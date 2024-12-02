# --------------------------------------------------------------------
#       Choose the mid-line of the original SDO (like Carrington do)
#       Author: Zhe Gao
#       Date: 2024-12-02
# --------------------------------------------------------------------

from typing import Optional
from datetime import datetime, timedelta
import os
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Create Carrington rotation movie')
    parser.add_argument('--input_dir', type=str, default="/mnt/nas/home/huxing/202407/nas/data/hmi/magnet_pt", help='Input directory path')
    parser.add_argument('--output_dir', type=str, default="/mnt/nas-tq/tianwen/home/gaozhe/SDO_data/hmi_1h_line/magnet_pt/", help='Output directory path')
    parser.add_argument('--num_cpus', type=int, default=64, help='Number of CPUs to use (default: 64)')
    return parser.parse_args()


def sel_mid_line(input_dir: str, output_dir: str, date_time: datetime) -> Optional[str]:
    """
    Select the mid-line of the solar data.

    Args:
        input_dir (str): Input directory.
        output_dir (str): Output directory.
        date_time (datetime): The datetime object to process.

    Returns:
        Optional[str]: The output file path if successful, otherwise None.
    """
    try:
        # Construct file paths
        relevant_path = f"{date_time.strftime('/%Y/%m/%d/hmi.M_720s.%Y%m%d_%H%M%S')}_TAI.pt"
        data_path = os.path.join(input_dir, relevant_path)
        output_path = os.path.join(output_dir, f"{date_time.strftime('/%Y/%m/%d/hmi.M_720s.%Y%m%d_%H%M%S')}_midline.pt")

        # Load and process data
        data = torch.load(data_path)
        mid_line_data = data[:, 512]  # Select mid-line

        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(mid_line_data, output_path)

        print(f"Processed {date_time} successfully.")
        return output_path

    except FileNotFoundError:
        print(f"No data found locally for {date_time}, skipping...")
    except Exception as e:
        print(f"Error processing {date_time}: {e}")

    return None


def main():
    """
    Main function to process the solar data in parallel.
    """
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    num_cpus = args.num_cpus

    # Define time range
    start_date = datetime(2010, 5, 1, 0, 0, 0)
    end_date = datetime(2024, 5, 1, 0, 0, 0)
    delta = timedelta(hours=1)

    # Generate list of datetime objects
    data_list = [start_date + i * delta for i in range(int((end_date - start_date) / delta))]

    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        futures = {
            executor.submit(sel_mid_line, input_dir, output_dir, date): date for date in data_list
        }

        for future in as_completed(futures):
            date = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"Output saved to {result}")
            except Exception as e:
                print(f"Failed to process {date}: {e}")


if __name__ == "__main__":
    main()
