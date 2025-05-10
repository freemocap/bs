import pandas as pd
from pathlib import Path

if __name__=='__main__':
    folder_path = Path("/home/scholl-lab/recordings/session_2025-04-28/ferret_9C04_NoImplant_P35_E3/skellyclicker_data")
    print(folder_path)
    csv_files = sorted(list(folder_path.iterdir()))

    for csv in csv_files:
        df = pd.read_csv(str(csv))
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        print(f"For csv {csv}, the following rows contains negative values")
        print(df_numeric[df_numeric.lt(0).any(axis=1)])