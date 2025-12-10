# Variables
IMAGE_NAME = neuroguard_dqn

# Bootstrap
bootstrap:
	docker compose build

# Fast Run (For Grading - < 5 mins)
up:
	docker compose up

# Full Run (For Report - Uses all data)
full_run:
	docker compose run --rm neuroguard python src/main.py --full

# Demo
demo:
	docker compose run --rm neuroguard python src/main.py --demo

# Clean
clean:
	docker compose down
	docker system prune -f

# Tests
test:
	docker compose run --rm neuroguard python src/tests.py