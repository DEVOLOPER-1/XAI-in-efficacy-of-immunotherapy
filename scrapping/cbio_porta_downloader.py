import logging
import os
import shutil
from datetime import datetime, timezone
from urllib.parse import urlparse

import polars as pl
from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

# ── Directories ───────────────────────────────────────────────────────────────
DEST_DIR = 'downloads_valid'
BAD_DIR = 'downloads_bad'
TEMP_DIR = 'downloads_tmp'
LOG_DIR = 'logs'

# ── Scraping config ───────────────────────────────────────────────────────────
TARGET_URL = 'https://www.cbioportal.org/datasets'
SEARCH_TERM = 'tcga'
CLICK_SELECTOR = 'a:has(i.fa-download)'
DELAY_BETWEEN_CLICKS = 1_200  # ms
BROWSER_HEADLESS = False
NAV_TIMEOUT = 60_000  # ms
DOWNLOAD_TIMEOUT = 180_000  # ms
SEARCH_TIMEOUT = 20_000  # ms
MIN_VALID_SIZE = 1_024  # bytes


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logger(name: str = 'downloader') -> logging.Logger:
	"""
	Configure and return a logger that writes to both stdout and a
	timestamped log file under LOG_DIR.
	"""
	os.makedirs(LOG_DIR, exist_ok=True)
	stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_path = os.path.join(LOG_DIR, f'{name}_{stamp}.log')

	fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
	datefmt = '%Y-%m-%d %H:%M:%S'

	logging.basicConfig(
		level=logging.DEBUG,
		format=fmt,
		datefmt=datefmt,
		handlers=[
			logging.FileHandler(log_path, encoding='utf-8'),
			logging.StreamHandler(),
		],
	)
	logger = logging.getLogger(name)
	logger.info('Log file: %s', log_path)
	return logger


log = setup_logger()


# ── File utilities ────────────────────────────────────────────────────────────


def make_unique_path(directory: str, filename: str) -> str:
	"""Return a non-colliding file path inside the given directory."""
	base, ext = os.path.splitext(filename)
	candidate = os.path.join(directory, filename)
	idx = 1
	while os.path.exists(candidate):
		candidate = os.path.join(directory, f'{base}_{idx}{ext}')
		idx += 1
	return candidate


def is_gzip(path: str) -> bool:
	"""Return True if the file starts with the gzip magic bytes."""
	try:
		with open(path, 'rb') as f:
			return f.read(2) == b'\x1f\x8b'
	except Exception:
		return False


def ensure_tar_gz(filename: str) -> str:
	"""Append .tar.gz to filename if the extension is missing."""
	return filename if filename.lower().endswith('.tar.gz') else filename + '.tar.gz'


def resolve_filename(suggested: str, url: str, index: int) -> str:
	"""Derive a local filename from the suggested name, URL path, or index fallback."""
	if suggested:
		return suggested
	segment = os.path.basename(urlparse(url).path)
	return segment or f'download_{index}'


# ── Link scraping ─────────────────────────────────────────────────────────────


def scrape_anchor_links(page: Page) -> pl.DataFrame:
	"""
	Collect all download anchor hrefs visible on the page before any clicks.
	Returns a Polars DataFrame with one row per anchor.
	"""
	anchors = page.locator(CLICK_SELECTOR)
	total = anchors.count()
	log.info('Scraping %d anchor link(s) from page.', total)

	records = []
	for i in range(total):
		anchor = anchors.nth(i)
		href = anchor.get_attribute('href') or ''
		visible = anchor.is_visible()
		records.append({
			'anchor_index': i + 1,
			'parsed_href': href,
			'visible': visible,
		})
		log.debug('Anchor %d | visible=%-5s | href=%s', i + 1, visible, href or '(none)')

	return pl.DataFrame(
		records,
		schema={
			'anchor_index': pl.Int32,
			'parsed_href': pl.String,
			'visible': pl.Boolean,
		},
	)


# ── Download processing ───────────────────────────────────────────────────────


def validate_and_move(tmp_path: str, size: int) -> tuple[str, str]:
	"""
	Validate the downloaded file and route it to DEST_DIR or BAD_DIR.
	Returns (final_path, status_label).
	"""
	is_valid = is_gzip(tmp_path) and size > MIN_VALID_SIZE

	if not is_valid:
		dest = make_unique_path(BAD_DIR, os.path.basename(tmp_path))
		shutil.move(tmp_path, dest)
		log.warning('INVALID  | size=%d | moved to BAD_DIR -> %s', size, dest)
		return dest, 'BAD'

	final_name = ensure_tar_gz(os.path.basename(tmp_path))
	dest = make_unique_path(DEST_DIR, final_name)
	shutil.move(tmp_path, dest)

	renamed = final_name != os.path.basename(tmp_path)
	status = 'OK (renamed)' if renamed else 'OK'
	log.info('VALID    | size=%d | status=%s | saved -> %s', size, status, dest)
	return dest, status


def process_anchor(page: Page, anchor, index: int, total: int) -> dict:
	"""
	Click a single download anchor, persist the file, validate it,
	and return a structured report record.
	"""
	started_at = datetime.now(timezone.utc)

	if not anchor.is_visible():
		log.warning('[%d/%d] Anchor not visible — skipping.', index, total)
		return _base_record(index, started_at, 'SKIPPED')

	log.info('[%d/%d] Clicking anchor...', index, total)

	try:
		with page.expect_download(timeout=DOWNLOAD_TIMEOUT) as dl_info:
			anchor.click()
		download = dl_info.value
		finished_at = datetime.now(timezone.utc)

		url = download.url
		suggested = download.suggested_filename or ''
		log.info('  URL       : %s', url)
		log.info('  Suggested : %s', suggested or '(none)')

		tmp_name = resolve_filename(suggested, url, index)
		tmp_path = make_unique_path(TEMP_DIR, tmp_name)
		download.save_as(tmp_path)
		size = os.path.getsize(tmp_path)

		final_path, status = validate_and_move(tmp_path, size)

		return {
			'anchor_index': index,
			'download_url': url,
			'suggested_name': suggested,
			'saved_path': final_path,
			'saved_dir': os.path.dirname(final_path),
			'filename': os.path.basename(final_path),
			'size_bytes': size,
			'is_gzip': is_gzip(final_path),
			'status': status,
			'started_at': started_at.isoformat(),
			'finished_at': finished_at.isoformat(),
			'elapsed_s': round((finished_at - started_at).total_seconds(), 2),
		}

	except PlaywrightTimeout:
		log.error('[%d/%d] Timeout waiting for download.', index, total)
		return _base_record(index, started_at, 'TIMEOUT')
	except Exception as exc:
		log.error('[%d/%d] Unexpected error: %s', index, total, exc)
		return _base_record(index, started_at, f'ERROR: {exc}')


def _base_record(index: int, started_at: datetime, status: str) -> dict:
	"""Minimal record for anchors that were skipped or failed before downloading."""
	return {
		'anchor_index': index,
		'download_url': '',
		'suggested_name': '',
		'saved_path': '',
		'saved_dir': '',
		'filename': '',
		'size_bytes': 0,
		'is_gzip': False,
		'status': status,
		'started_at': started_at.isoformat(),
		'finished_at': '',
		'elapsed_s': 0.0,
	}


# ── Browser helpers ───────────────────────────────────────────────────────────


def apply_search_filter(page: Page, term: str) -> None:
	"""Type into the table search input to filter results."""
	try:
		field = page.get_by_label('Table Search Input')
		field.wait_for(state='visible', timeout=SEARCH_TIMEOUT)
		field.fill(term)
		page.wait_for_timeout(1_000)
		log.info('Search filter applied: %r', term)
	except PlaywrightTimeout:
		log.warning('Search field not found; continuing without filtering.')


# ── Polars reporting & validation ─────────────────────────────────────────────


def build_report_dataframe(records: list[dict]) -> pl.DataFrame:
	"""Cast the raw report records into a typed Polars DataFrame."""
	return pl.DataFrame(
		records,
		schema={
			'anchor_index': pl.Int32,
			'download_url': pl.String,
			'suggested_name': pl.String,
			'saved_path': pl.String,
			'saved_dir': pl.String,
			'filename': pl.String,
			'size_bytes': pl.Int64,
			'is_gzip': pl.Boolean,
			'status': pl.String,
			'started_at': pl.String,
			'finished_at': pl.String,
			'elapsed_s': pl.Float64,
		},
	)


def validate_links_vs_downloads(
	links_df: pl.DataFrame,
	report_df: pl.DataFrame,
) -> pl.DataFrame:
	"""
	Left-join parsed anchor hrefs against actual download URLs.
	Adds a `matched` boolean column — True when the download URL contains
	the tar.gz filename extracted from the pre-click parsed href.
	"""
	return links_df.join(
		report_df.select(['anchor_index', 'download_url', 'filename', 'status', 'size_bytes']),
		on='anchor_index',
		how='left',
	).with_columns(
		pl
		.when(
			pl.col('parsed_href').is_not_null()
			& pl.col('download_url').is_not_null()
			& (pl.col('parsed_href') != '')
			& (pl.col('download_url') != '')
			& pl.col('download_url').str.contains(
				pl.col('parsed_href').str.extract(r'([^/]+\.tar\.gz)', 0).fill_null('')
			)
		)
		.then(True)
		.otherwise(False)
		.alias('matched')
	)


def export_reports(
	report_df: pl.DataFrame,
	validation_df: pl.DataFrame,
	stamp: str,
) -> None:
	"""Write the full report and the link-validation table as CSV files under LOG_DIR."""
	report_path = os.path.join(LOG_DIR, f'report_{stamp}.csv')
	validation_path = os.path.join(LOG_DIR, f'validation_{stamp}.csv')

	report_df.write_csv(report_path)
	validation_df.write_csv(validation_path)

	log.info('Report CSV     -> %s', report_path)
	log.info('Validation CSV -> %s', validation_path)


def log_summary_tables(report_df: pl.DataFrame, validation_df: pl.DataFrame) -> None:
	"""Log human-readable tabular summaries of status, directories, and link matching."""
	status_counts = (
		report_df.group_by('status').agg(pl.len().alias('count')).sort('count', descending=True)
	)

	dir_counts = (
		report_df
		.filter(pl.col('saved_dir') != '')
		.group_by('saved_dir')
		.agg(
			pl.len().alias('files'),
			pl.col('size_bytes').sum().alias('total_bytes'),
		)
		.sort('files', descending=True)
	)

	matched_count = validation_df.filter(pl.col('matched')).height
	unmatched = validation_df.filter(~pl.col('matched'))
	validation_view = validation_df.select([
		'anchor_index',
		'parsed_href',
		'filename',
		'status',
		'matched',
	])

	log.info('\n%s\n  STATUS BREAKDOWN\n%s\n%s', '─' * 60, '─' * 60, status_counts)
	log.info('\n%s\n  FILES PER DIRECTORY\n%s\n%s', '─' * 60, '─' * 60, dir_counts)
	log.info(
		'\n%s\n  LINK -> FILE VALIDATION  (matched=%d / total=%d)\n%s\n%s',
		'─' * 60,
		matched_count,
		validation_df.height,
		'─' * 60,
		validation_view,
	)

	if unmatched.height:
		log.warning(
			'%d anchor(s) did not match their parsed href:\n%s',
			unmatched.height,
			unmatched.select(['anchor_index', 'parsed_href', 'download_url']),
		)


# ── Entry point ───────────────────────────────────────────────────────────────


def run() -> None:
	"""Orchestrate navigation, scraping, downloading, validation, and reporting."""
	stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

	for directory in (DEST_DIR, BAD_DIR, TEMP_DIR, LOG_DIR):
		os.makedirs(directory, exist_ok=True)

	records: list[dict]

	with sync_playwright() as pw:
		browser = pw.chromium.launch(headless=BROWSER_HEADLESS, downloads_path=TEMP_DIR)
		context = browser.new_context(accept_downloads=True)
		page = context.new_page()

		log.info('Navigating to %s', TARGET_URL)
		page.goto(TARGET_URL, wait_until='networkidle', timeout=NAV_TIMEOUT)

		if SEARCH_TERM:
			apply_search_filter(page, SEARCH_TERM)

		links_df = scrape_anchor_links(page)
		anchors = page.locator(CLICK_SELECTOR)
		total = anchors.count()
		log.info('Starting download loop — %d anchor(s) found.', total)

		records = [
			process_anchor(page, anchors.nth(i), i + 1, total)
			for i in range(total)
			if not page.wait_for_timeout(DELAY_BETWEEN_CLICKS) or True
		]

		log.info('Download loop complete. Closing browser.')
		browser.close()

	report_df = build_report_dataframe(records)
	validation_df = validate_links_vs_downloads(links_df, report_df)

	log_summary_tables(report_df, validation_df)
	export_reports(report_df, validation_df, stamp)


if __name__ == '__main__':
	run()
