import pandas as pd

df = pd.read_csv(r"artifacts\real_task04_test\ingest\keyframe_mapping.csv")
v10 = df[df['video_id'] == 'L21_V010'].head(20)
v25 = df[df['video_id'] == 'L21_V025'].head(20)
subset = pd.concat([v10, v25])
subset.to_csv(r"artifacts\subset_mapping.csv", index=False)
print("Subset generated with 40 keyframes.")
