# Variables
IMAGE_NAME = neuroguard_dqn
CONTAINER_NAME = neuroguard_dqn

# Bootstrap: Build the container
bootstrap:
	docker-compose build

# Run the training/detection loop
up:
	docker-compose up

# Clean up containers
clean:
	docker-compose down
	docker system prune -f

# Demo: Run a pre-trained model or a quick evaluation loop
demo:
	docker-compose run --rm neuroguard python src/main.py --mode demo

# Download dataset (Helper target)
download-data:
	@echo "Please manually place NSL-KDD.csv in ./data/ for now."