import pygame
import numpy as np


pygame.init()
pygame.display.set_caption("Artificial Horizon")
size = width, height = 500, 500
bgColor = 255, 255, 255

screen = pygame.display.set_mode(size)
frame = pygame.image.load(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\AH_images\Frame.png").convert_alpha()
inel = pygame.image.load(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\AH_images\Ring.png").convert_alpha()
interior = pygame.image.load(r"C:\Users\edoua\Documents\Birse\Bristol\MSc Thesis\drone_volcanic_monitoring\Drone_Volcanic_Monitoring\AH_images\Interior.png").convert_alpha()

font=pygame.font.Font(None, 60)
text = font.render("Waiting for flight logs",1,(255,255,255))
screen.blit(text, (30, 300))
pygame.display.flip()

def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect)

def draw_window(set_pitch, set_roll, scale_factor = 1):
    screen.fill(bgColor)
    blitRotateCenter(screen, interior, (-width/2, -height/2 + set_pitch*scale_factor), set_roll)
    blitRotateCenter(screen, inel, (0, 0), set_roll)
    screen.blit(frame, (0,0))
    pygame.display.update()



def artificial_horizon_play(set_pitch, set_roll):
        scale_factor = 7
        draw_window(set_pitch, set_roll, scale_factor)




