# Variables
IMAGE_NAME = neuroguard_dqn
CONTAINER_NAME = neuroguard_dqn

# Bootstrap: Build the container
bootstrap:
	docker-compose build

# Run the training loop (This is "make up")
up:
	docker-compose up

# Run the DEMO mode (This is "make demo")
demo:
	docker-compose run --rm neuroguard python src/main.py --demo

# Clean up
clean:
	docker-compose down
	docker system prune -f

# Run Tests
test:
	docker-compose run --rm neuroguard python src/tests.py