from clip_video_processor.multi_view_summarizer import MultiViewSummarizer

# Initialize the summarizer
summarizer = MultiViewSummarizer(config_path='config.json')

# Process videos to store embeddings
video_paths = ['path_to_your_video1.mp4', 'path_to_your_video2.mp4']  # Add more video paths as needed
summarizer.process_videos(video_paths)

# Generate a summary for a query
query_text = "A person playing guitar"
video_path = video_paths[0]  # Path to the video for visualization
summary = summarizer.generate_summary(query_text, video_path)

# Visualize the summary and timeline
summarizer.visualize_summary(summary, video_path)

# Close the Qdrant connection
summarizer.close()
