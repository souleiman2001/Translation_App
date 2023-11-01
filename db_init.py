from app import app, db  # Importing app as well from app.py


def init_db():
    with app.app_context():
        db.create_all()

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully!")
