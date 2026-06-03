import argparse
import logging
import sys

from dask.distributed import Client

from blackmarble_toolkit.workflows import (
    aggregate_workflow,
    download_workflow,
    preprocess_workflow,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Black Marble NTL Toolkit CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # DOWNLOAD
    download_parser = subparsers.add_parser(
        "download",
        help="Download Black Marble NTL data to a Zarr store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download_parser.add_argument(
        "--product",
        type=str,
        required=True,
        choices=["VNP46A1", "VNP46A2", "NOAA/VIIRS/DNB/ANNUAL_V22"],
        help="The Black Marble product to download",
    )
    download_parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    download_parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Path to a GeoJSON or Shapefile defining the region",
    )
    download_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output Zarr store path (e.g., raw.zarr)",
    )
    download_parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        help="Optional list of bands to select",
    )
    download_parser.add_argument(
        "--scale",
        type=float,
        help="Spatial resolution in degrees (defaults to native scale)",
    )
    download_parser.add_argument(
        "--chunks",
        type=str,
        default="auto",
        help="Chunk size/scheme for the resulting dataset",
    )

    # PREPROCESS
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Preprocess Black Marble data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    preprocess_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input Zarr store",
    )
    preprocess_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML/JSON configuration file defining the preprocessing steps",
    )
    preprocess_parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output Zarr store path (e.g., preprocessed.zarr)",
    )

    # AGGREGATE
    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate preprocessed data over vector geometries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    aggregate_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the preprocessed Zarr store",
    )
    aggregate_parser.add_argument(
        "--region",
        type=str,
        required=True,
        help="Path to a GeoJSON or Shapefile defining the aggregation regions",
    )
    aggregate_parser.add_argument(
        "--geo-id",
        type=str,
        default="geonameid",
        help="Column name in the region file that uniquely identifies each shape",
    )
    aggregate_parser.add_argument(
        "--out-zarr",
        type=str,
        required=True,
        help="Output Zarr store path for the aggregated dataset",
    )
    aggregate_parser.add_argument(
        "--out-csv",
        type=str,
        help="Optional output CSV path for the aggregated data",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    client = Client()
    logger.info(
        f"Dask distributed client initialized. Dashboard URL: {client.dashboard_link}"
    )

    try:
        if args.command == "download":
            download_workflow(
                product=args.product,
                start_date=args.start_date,
                end_date=args.end_date,
                region_file=args.region,
                out_zarr=args.out,
                bands=args.bands,
                scale=args.scale,
                chunks=args.chunks,
            )
        elif args.command == "preprocess":
            preprocess_workflow(
                input_zarr=args.input,
                config_file=args.config,
                out_zarr=args.out,
            )
        elif args.command == "aggregate":
            aggregate_workflow(
                input_zarr=args.input,
                region_file=args.region,
                geo_id_col=args.geo_id,
                out_zarr=args.out_zarr,
                out_csv=args.out_csv,
            )
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"CLI workflow failed: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
