import pygame
import random
import sys

pygame.init()

WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rainy Background")
clock = pygame.time.Clock()

BACKGROUND_COLOR = (20, 30, 50)
RAIN_COLOR = (200, 200, 200)
CLOUD_COLOR = (70, 70, 80)

class Raindrop:
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(-50, 0)
        self.length = random.randint(10, 20)
        self.speed = random.uniform(5, 15)
        self.thickness = 1

    def update(self):
        self.y += self.speed
        if self.y > HEIGHT:
            self.x = random.randint(0, WIDTH)
            self.y = random.randint(-50, 0)

    def draw(self, surface):
        pygame.draw.line(
            surface,
            RAIN_COLOR,
            (self.x, self.y),
            (self.x, self.y + self.length),
            self.thickness
        )

class Cloud:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.speed = random.uniform(0.5, 2)

    def update(self):
        self.x += self.speed
        if self.x > WIDTH + 100:
            self.x = -100

    def draw(self, surface):
        circle1_x = self.x
        circle2_x = self.x + self.size // 2
        circle3_x = self.x + self.size
        
        pygame.draw.circle(surface, CLOUD_COLOR, (int(circle1_x), int(self.y)), int(self.size // 3))
        pygame.draw.circle(surface, CLOUD_COLOR, (int(circle2_x), int(self.y - self.size // 4)), int(self.size // 2))
        pygame.draw.circle(surface, CLOUD_COLOR, (int(circle3_x), int(self.y)), int(self.size // 3))

raindrops = [Raindrop() for _ in range(200)]
clouds = [
    Cloud(100, 50, 60),
    Cloud(400, 80, 80),
    Cloud(800, 60, 70),
    Cloud(1100, 100, 75)
]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND_COLOR)
    
    for cloud in clouds:
        cloud.update()
        cloud.draw(screen)

    for raindrop in raindrops:
        raindrop.update()
        raindrop.draw(screen)

    clock.tick(60)
    pygame.display.flip()

pygame.quit()
sys.exit()
