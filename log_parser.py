import re
import json

def parse_log_file(log_file_path):
    """
    Parse the log file containing video processing information and extract segment data.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        list: List of dictionaries containing video information and segments
    """
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # Split by video entries (starting with [X/50] Processing)
    video_entries = re.split(r'\[\d+/\d+\] Processing:', log_content)[1:]
    
    videos = []
    
    for entry in video_entries:
        # Extract filename
        filename_match = re.search(r'raw_videos_720p/(.*?\.mp4)', entry)
        if not filename_match:
            continue
        
        filename = filename_match.group(1)
        
        # Find all segment information
        segment_frames = re.findall(r'Segment \d+: frames (\d+)-(\d+) \(([\d.]+) seconds\)', entry)
        
        # Find all effect applications
        effect_applications = re.findall(r'Segment \d+: frames (\d+)-(\d+).*?\n(.*?)to segment \d+', entry, re.DOTALL)
        
        segments = []
        
        # Create segment data structure
        for i, (start_frame, end_frame, duration) in enumerate(segment_frames):
            # Find the corresponding effect for this segment
            effect = None
            for app_start, app_end, app_text in effect_applications:
                if app_start == start_frame and app_end == end_frame:
                    effect_match = re.search(r'Applying effect: ([\w_]+)', app_text)
                    if effect_match:
                        effect = effect_match.group(1)
                    break
            
            # If no effect found, check if "No effect applied" is mentioned
            if effect is None:
                no_effect_match = re.search(
                    rf'Segment \d+: frames {start_frame}-{end_frame}.*?\nNo effect applied to segment \d+ \(random chance\)', 
                    entry
                )
                if not no_effect_match:
                    # Look for the effect outside of the pattern in case the format is different
                    segment_effect_match = re.search(
                        rf'Segment \d+: frames {start_frame}-{end_frame}.*?\nApplying effect: ([\w_]+) to segment \d+',
                        entry
                    )
                    if segment_effect_match:
                        effect = segment_effect_match.group(1)
            
            segments.append({
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "duration": float(duration),
                "effect": 1 if effect is not None else 0
            })
        
        videos.append({
            "filename": filename,
            "segments": segments
        })
    
    return videos

def save_videos_to_json(videos, output_path):
    """
    Save the parsed video data to a JSON file
    
    Args:
        videos (list): List of video dictionaries
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w') as file:
        json.dump(videos, file, indent=4)

def save_videos_to_python(videos, output_path):
    """
    Save the parsed video data as a Python list assignment
    
    Args:
        videos (list): List of video dictionaries
        output_path (str): Path to save the Python file
    """
    with open(output_path, 'w') as file:
        file.write("# Parse the log file data\nvideos = [\n")
        
        for i, video in enumerate(videos):
            file.write("    {\n")
            file.write(f'        "filename": "{video["filename"]}",\n')
            file.write('        "segments": [\n')
            
            for j, segment in enumerate(video["segments"]):
                file.write("            {")
                file.write(f'"start_frame": {segment["start_frame"]}, ')
                file.write(f'"end_frame": {segment["end_frame"]}, ')
                file.write(f'"duration": {segment["duration"]}, ')
                
                file.write(f'"effect": {segment["effect"]}')
                
                if j < len(video["segments"]) - 1:
                    file.write("},\n")
                else:
                    file.write("}\n")
            
            if i < len(videos) - 1:
                file.write("        ]\n    },\n")
            else:
                file.write("        ]\n    }\n")
        
        file.write("]\n")

def main():
    # Specify the path to your log file
    log_file_path = "log_files/log_prepro"  # Change this to your actual log file path
    
    # Parse the log file
    videos = parse_log_file(log_file_path)
    
    # Save the parsed data
    save_videos_to_json(videos, "video_effects_data.json")
    save_videos_to_python(videos, "video_effects_data.py")
    
    print(f"Successfully parsed {len(videos)} videos with their segments.")
    print("JSON data saved to 'video_effects_data.json'")
    print("Python data saved to 'video_effects_data.py'")

if __name__ == "__main__":
    main()