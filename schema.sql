-- Generated from SQLAlchemy models in src/pm_insights/db.py.

-- Regenerate with: python scripts/create_schema.py --output schema.sql


CREATE TABLE analysis_feedback (
	id SERIAL NOT NULL, 
	user_id INTEGER NOT NULL, 
	meeting_id VARCHAR(64) NOT NULL, 
	item_type VARCHAR(64) NOT NULL, 
	source_text TEXT NOT NULL, 
	predicted_label VARCHAR(64) NOT NULL, 
	corrected_label VARCHAR(64) NOT NULL, 
	corrected_text TEXT NOT NULL, 
	metadata_json JSON NOT NULL, 
	used_for_training BOOLEAN NOT NULL, 
	training_run_id VARCHAR(128) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE analysis_results (
	id SERIAL NOT NULL, 
	meeting_id VARCHAR(64) NOT NULL, 
	tasks_json JSON NOT NULL, 
	questions_answers_json JSON NOT NULL, 
	decisions_json JSON NOT NULL, 
	deadlines_json JSON NOT NULL, 
	responsibles_json JSON NOT NULL, 
	sentiment_json JSON NOT NULL, 
	aspects_json JSON NOT NULL, 
	topics_json JSON NOT NULL, 
	metrics_json JSON NOT NULL, 
	result_json JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE meetings (
	id SERIAL NOT NULL, 
	meeting_id VARCHAR(64) NOT NULL, 
	user_id INTEGER, 
	meeting_title VARCHAR(255) NOT NULL, 
	meeting_key VARCHAR(512) NOT NULL, 
	project_name VARCHAR(255) NOT NULL, 
	meeting_date VARCHAR(32), 
	participants TEXT NOT NULL, 
	source_audio TEXT NOT NULL, 
	original_filename TEXT NOT NULL, 
	stored_filename TEXT NOT NULL, 
	file_extension VARCHAR(16) NOT NULL, 
	file_size_bytes BIGINT NOT NULL, 
	upload_date TIMESTAMP WITH TIME ZONE NOT NULL, 
	processing_status VARCHAR(32) NOT NULL, 
	metadata_json JSON NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE processing_logs (
	id SERIAL NOT NULL, 
	meeting_id VARCHAR(64) NOT NULL, 
	step VARCHAR(64) NOT NULL, 
	status VARCHAR(32) NOT NULL, 
	message TEXT NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE transcripts (
	id SERIAL NOT NULL, 
	meeting_id VARCHAR(64) NOT NULL, 
	text TEXT NOT NULL, 
	segments_json JSON NOT NULL, 
	language VARCHAR(16) NOT NULL, 
	asr_model VARCHAR(128) NOT NULL, 
	duration_seconds FLOAT NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);


CREATE TABLE users (
	id SERIAL NOT NULL, 
	email VARCHAR(255) NOT NULL, 
	username VARCHAR(64) NOT NULL, 
	full_name VARCHAR(255) NOT NULL, 
	password_hash TEXT NOT NULL, 
	is_active BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id)
);

CREATE INDEX ix_analysis_feedback_meeting_id ON analysis_feedback (meeting_id);

CREATE INDEX ix_analysis_feedback_user_id ON analysis_feedback (user_id);

CREATE INDEX ix_analysis_results_meeting_id ON analysis_results (meeting_id);

CREATE UNIQUE INDEX ix_meetings_meeting_id ON meetings (meeting_id);

CREATE INDEX ix_meetings_meeting_key ON meetings (meeting_key);

CREATE INDEX ix_meetings_user_id ON meetings (user_id);

CREATE INDEX ix_processing_logs_meeting_id ON processing_logs (meeting_id);

CREATE INDEX ix_transcripts_meeting_id ON transcripts (meeting_id);

CREATE UNIQUE INDEX ix_users_email ON users (email);

CREATE UNIQUE INDEX ix_users_username ON users (username);
