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

def save_words_to_dictionary(dictionary_id, detected_terms):
    """Save detected terms as words in the words table"""
    if not detected_terms or not isinstance(detected_terms, dict):
        return
    
    terms_list = detected_terms.get('terms', [])
    if not terms_list:
        return
    
    with get_db() as conn:
        with conn.cursor() as cursor:
            for term in terms_list:
                if isinstance(term, dict):
                    word = term.get('word', '').strip()
                    pronunciation = term.get('pronunciation_japanese', '').strip()
                    meaning = term.get('meaning_japanese', '').strip()
                    why_important = term.get('why_important', '').strip()
                    
                    if word:
                        # Check if word already exists for this dictionary
                        cursor.execute("""
                            SELECT id FROM words 
                            WHERE dictionary_id = %s AND word = %s
                        """, (dictionary_id, word))
                        existing = cursor.fetchone()
                        
                        if not existing:
                            # Insert new word
                            try:
                                cursor.execute("""
                                    INSERT INTO words (dictionary_id, word, pronunciation_japanese, meaning_japanese, why_important)
                                    VALUES (%s, %s, %s, %s, %s)
                                """, (dictionary_id, word, pronunciation, meaning, why_important))
                            except Exception as e:
                                # Ignore duplicate errors
                                pass
            conn.commit()