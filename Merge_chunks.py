import os
import json
import math 

n = 8

for filename in os.listdir("jsons"):
    if filename.endswith("json"):
        file_path = os.path.join("jsons", filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            total_chunks = len(data['chunks'])
            num_groups = math.ceil(total_chunks/n)

            for i in range(num_groups):
                start_idx = i*n
                end_idx = min((i+1)*n, total_chunks)
                chunk_group = data['chunks'][start_idx:end_idx]

                new_chunks.append({
                    "number" : data['chunks'][0]['number'],
                    "title" : data['chunks'][0]['title'],
                    "start" : chunk_group[0]['start'],
                    "end" : chunk_group[-1]['end'],
                    "text" : " ".join([chunk['text'] for chunk in chunk_group])
                })
            #save file without double .json
            os.makedirs("newjsons", exist_ok=True)
            with open(os.path.join("newjsons", filename), "w", encoding="utf-8") as json_file:
                json.dump({"chunks": new_chunks, "text": data['text']}, json_file)

# import os
# import json

# MAX_GAP = 8          # seconds between chunks
# MAX_DURATION = 40    # max merged chunk length


# def merge_chunks(chunks):
#     merged = []
#     current = chunks[0].copy()

#     for next_chunk in chunks[1:]:
#         same_video = next_chunk["title"] == current["title"]
#         gap = next_chunk["start"] - current["end"]
#         duration = next_chunk["end"] - current["start"]

#         if same_video and gap <= MAX_GAP and duration <= MAX_DURATION:
#             # merge
#             current["end"] = next_chunk["end"]
#             current["text"] += " " + next_chunk["text"]
#         else:
#             merged.append(current)
#             current = next_chunk.copy()

#     merged.append(current)
#     return merged


# # Process all JSON files
# os.makedirs("newjsons", exist_ok=True)

# for filename in os.listdir("jsons"):
#     if filename.endswith(".json"):
#         file_path = os.path.join("jsons", filename)

#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 data = json.load(f)
#         except json.JSONDecodeError as e:
#             print(f"Skipping {filename}: {e}")
#             continue

#         chunks = data["chunks"]

#         # Sort chunks (important)
#         chunks = sorted(chunks, key=lambda x: x["start"])

#         new_chunks = merge_chunks(chunks)

#         # Save
#         with open(os.path.join("newjsons", filename), "w", encoding="utf-8") as json_file:
#             json.dump({
#                 "chunks": new_chunks,
#                 "text": data.get("text", "")
#             }, json_file, indent=2)