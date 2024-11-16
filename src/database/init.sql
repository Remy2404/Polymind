CREATE TABLE IF NOT EXISTS users (
    user_id BIGINT PRIMARY KEY,
    joined_date TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    messages_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    markdown_enabled BOOLEAN DEFAULT TRUE,
    code_suggestions BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_contexts (
    id SERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(user_id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant'))
);

CREATE TABLE IF NOT EXISTS user_stats (
    user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    total_messages INTEGER DEFAULT 0,
    total_images INTEGER DEFAULT 0,
    last_interaction TIMESTAMP
);

CREATE INDEX idx_user_contexts_user_id ON user_contexts(user_id);
CREATE INDEX idx_user_contexts_created_at ON user_contexts(created_at); 