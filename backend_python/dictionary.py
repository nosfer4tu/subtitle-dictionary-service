from database.connection import get_db

def add_dictionary_entry(user_id, video_id, title):
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
				"""
				INSERT INTO dictionaries (user_id, youtube_video_id, title)
				VALUES (%s, %s, %s)
				RETURNING id
				""",
				(user_id, video_id, title)
			)
            new_id = cursor.fetchone()[0]
            conn.commit()
            return new_id

def update_dictionary_transcription(id, transcription, translation):
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
				"""
				UPDATE dictionaries 
				SET transcription = %s, translation = %s
				WHERE id = %s
                """,
                (transcription, translation, id)
			)
            conn.commit()