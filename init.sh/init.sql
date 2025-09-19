# Create with a different name
cat > 01-init.sql << 'EOF'
-- 1. Buat role ryan (hanya jika belum ada)
DO
$do$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='ryan') THEN
      CREATE ROLE ryan LOGIN PASSWORD 'password';
   END IF;
END
$do$;

-- 2. Buat database chatbot_db (hanya dijalankan di volume baru)
SELECT 'CREATE DATABASE chatbot_db OWNER ryan'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'chatbot_db')\gexec

-- 3. Kasih privilege ke ryan
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO ryan;
EOF