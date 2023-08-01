import streamlit as st
from main import *

def main():
    st.title("Ai Powered Tool for Technical Learning")
    
    # Input for YouTube URL
    youtube_url = st.text_input("Enter the YouTube URL:", "")
    
    # Input for OpenAI API key
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    
    if st.button("Generate Summaries"):
        if youtube_url and api_key:
            video_id = extract_video_id(youtube_url)
            transcript_text = get_youtube_transcript(video_id)
            summary = summarize_transcript(transcript_text)
            st.subheader("Summary:")
            st.write(summary)
            
    if st.button("Generate Steps"):
        if youtube_url and api_key:
            video_id = extract_video_id(youtube_url)
            transcript_text = get_youtube_transcript(video_id)
            steps = generate_steps_from_text(transcript_text, api_key)
            st.subheader("Steps:")
            st.write(steps)
            
    if st.button("Generate Quiz Questions"):
        if youtube_url and api_key:
            video_id = extract_video_id(youtube_url)
            transcript_text = get_youtube_transcript(video_id)
            quiz_questions = generate_quiz_questions(transcript_text, api_key)
            st.subheader("Quiz Questions:")
            st.write(quiz_questions)

if __name__ == "__main__":
    main()
