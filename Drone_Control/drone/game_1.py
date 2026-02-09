import pygame
import time

# 设置窗口尺寸和颜色
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 620
BACKGROUND_COLOR = (255, 255, 255)

# 每个板块的宽度和颜色
leftWidth = 200
leftColor = (255, 245, 238)
middleWidth = 500
middleColor = (187, 255, 255)
rightWidth = 0
rightColor = (152, 245, 255)

# attention进度条
attentionWidth = 150
attentionHeight = 30
attentionColor = (255, 228, 225)

# meditation进度条
meditationWidth = 150
meditationHeight = 30
meditationColor = (255, 228, 225)

# 进度条细节
progressColor = (9, 190, 255)
# progressWidth = 12
progressHeight = 30

# 遮罩层
alpha_left = 150
alpha_right = 150
transparent_rect_left = pygame.Surface((500, 700), pygame.SRCALPHA)
transparent_rect_right = pygame.Surface((500, 700), pygame.SRCALPHA)
transparent_rect_left.fill((211, 211, 211, alpha_left))
transparent_rect_right.fill((211, 211, 211, alpha_right))


class Game:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("test")
        self.angle = 0
        self.height = 0
        x = leftWidth + middleWidth + rightWidth / 2
        y = WINDOW_HEIGHT / 2
        self.core = (x, y)

    def drawbk(self):
        self.window.fill(BACKGROUND_COLOR)
        pygame.draw.rect(self.window, leftColor, (0, 0, leftWidth, WINDOW_HEIGHT))
        pygame.draw.rect(self.window, middleColor, (leftWidth, 0, middleWidth, WINDOW_HEIGHT))
        pygame.draw.rect(self.window, rightColor, (leftWidth + middleWidth, 0, rightWidth, WINDOW_HEIGHT))
        pygame.draw.rect(self.window, attentionColor, (0, 100, attentionWidth, attentionHeight))
        pygame.draw.rect(self.window, meditationColor, (0, 300, meditationWidth, meditationHeight))

        # 加载图片
        img_drone = pygame.image.load("sources/drone.png")
        img_drone = pygame.transform.rotozoom(img_drone, 0, 0.5)  # 保持当前缩放比例
        # 调整图像大小
        desired_width = int(img_drone.get_width() * 0.5)  # 将图像宽度缩小为当前宽度的50%
        desired_height = int(img_drone.get_height() * 0.5)  # 将图像高度缩小为当前高度的50%
        img_drone = pygame.transform.scale(img_drone, (desired_width, desired_height))
        # 绘制竖直方向上的刻度尺
        scale_color = (0, 0, 0)  # 刻度尺颜色
        scale_x = leftWidth  # 刻度尺起始位置（在图像左边）
        scale_y_start = 20
        scale_y_end = 720
        scale_max = 250
        scale_min = 0
        for i in range(scale_y_start, scale_y_end, 175):  # 刻度间隔为20像素
            pygame.draw.line(self.window, scale_color, (scale_x, i), (scale_x + 20, i))
            # 显示刻度值标签
            scale_value = int(
                scale_max - ((i - scale_y_start) / (scale_y_end - scale_y_start)) * (200 - scale_min)-100)
            font = pygame.font.Font(None, 20)  # 设置字体和字号
            text = font.render(str(scale_value), True, scale_color)  # 创建显示刻度值的文本
            text_rect = text.get_rect()
            text_rect.center = (scale_x - 20, i)  # 文本位置
            self.window.blit(text, text_rect)  # 显示文本
        # 绘制缩小后的图像
        self.window.blit(img_drone, (
            leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
            WINDOW_HEIGHT  - img_drone.get_height()  - self.height*3))

        # 加载图片
        # img_droneLVL = pygame.image.load("sources/droneLVL.png")
        # img_droneLVL = pygame.transform.rotozoom(img_droneLVL, 0, 0.5)  # 保持当前缩放比例
        #
        # # 调整图像大小
        # desired_width = int(img_droneLVL.get_width() * 1)  # 将图像宽度缩小为当前宽度的50%
        # desired_height = int(img_droneLVL.get_height() * 1)  # 将图像高度缩小为当前高度的50%
        # img_droneLVL = pygame.transform.scale(img_droneLVL, (desired_width, desired_height))
        #
        # # 绘制缩小后的图像
        # self.window.blit(img_droneLVL, (leftWidth + middleWidth + rightWidth / 2 - img_droneLVL.get_width() / 2,
        #                                 WINDOW_HEIGHT / 2 - img_droneLVL.get_height() / 2))
        #
        # self.window.blit(img_droneLVL, img_droneLVL.get_rect(center=tuple(self.core)))

        ft_vertical = pygame.font.Font("sources/STZHONGS.TTF", 40)
        ft_level = pygame.font.Font("sources/STZHONGS.TTF", 40)
        ft_attention = pygame.font.Font("sources/STZHONGS.TTF", 25)
        ft_meditation = pygame.font.Font("sources/STZHONGS.TTF", 25)
        ft_height = pygame.font.Font("sources/STZHONGS.TTF", 20)
        ft_heightNumber = pygame.font.Font("sources/STZHONGS.TTF", 20)
        ft_angle = pygame.font.Font("sources/STZHONGS.TTF", 25)
        ft_angleNumber = pygame.font.Font("sources/STZHONGS.TTF", 25)

        text_vertical = ft_vertical.render("垂直", True, (0, 0, 0))
        text_level = ft_level.render("水平", True, (0, 0, 0))
        text_attention = ft_attention.render("集中程度", True, (0, 0, 0))
        text_meditation = ft_meditation.render("分散程度", True, (0, 0, 0))
        text_height = ft_height.render("当前高度", True, (0, 0, 0))
        text_heightNumber = ft_heightNumber.render(str(self.height) + "m", True, (0, 0, 0))
        text_angle = ft_angle.render("旋转角度", True, (0, 0, 0))
        text_angleNumber = ft_angleNumber.render(str(self.angle) + "°", True, (0, 0, 0))

        self.window.blit(text_vertical, (leftWidth + middleWidth / 2 - 40, 20))
        # self.window.blit(text_level, (leftWidth + middleWidth + rightWidth / 2 - 40, 20))
        self.window.blit(text_attention, (20, 50))
        self.window.blit(text_meditation, (20, 250))
        self.window.blit(text_height, (10, 450))
        self.window.blit(text_heightNumber, (100, 450))
        # self.window.blit(text_angle, (leftWidth + middleWidth + rightWidth / 2 - 100, 600))
        # self.window.blit(text_angleNumber, (leftWidth + middleWidth + rightWidth / 2 + 20, 600))

        # self.window.blit(transparent_rect_left, (leftWidth, 0))
        # self.window.blit(transparent_rect_right, (leftWidth + middleWidth, 0))

    def drawProgress(self, attention, meditation):

        pygame.draw.rect(self.window, progressColor, (0, 100, attention, progressHeight))
        pygame.draw.rect(self.window, progressColor, (0, 300, meditation, progressHeight))
        ft_attentionNumber = pygame.font.Font("sources/STZHONGS.TTF", 20)
        ft_meditationNumber = pygame.font.Font("sources/STZHONGS.TTF", 20)
        text_attentionNumber = ft_attentionNumber.render(str(attention), True, (0, 0, 0))
        text_meditationNumber = ft_meditationNumber.render(str(meditation), True, (0, 0, 0))
        self.window.blit(text_attentionNumber, (140, 55))
        self.window.blit(text_meditationNumber, (140, 255))
        pygame.display.update()

    def switchUpSystem(self):
        alpha_left = 0
        alpha_right = 150
        transparent_rect_left.fill((211, 211, 211, alpha_left))
        transparent_rect_right.fill((211, 211, 211, alpha_right))
        pygame.display.update()

    def switchLevelSystem(self):
        alpha_left = 150
        alpha_right = 0
        transparent_rect_left.fill((211, 211, 211, alpha_left))
        transparent_rect_right.fill((211, 211, 211, alpha_right))
        pygame.display.update()

    def Up(self, height):
        # img_uparrow = pygame.image.load("sources/uparrow.png")
        # img_uparrow = pygame.transform.rotozoom(img_uparrow, 0, 0.08)
        # img_uparrowWidth, img_uparrowHeight = img_uparrow.get_size()
        # self.window.blit(img_uparrow, (leftWidth + middleWidth / 2 - img_uparrowWidth / 2, 100))
        # pygame.display.update()

        img_drone = pygame.image.load("sources/drone.png")
        img_drone = pygame.transform.rotozoom(img_drone, 0, 0.25)  # 保持当前缩放比例
        # 调整图像大小
        self.window.blit(img_drone, (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                                     WINDOW_HEIGHT  -  img_drone.get_height()  - self.height*3))
        for i in range(height):
            pygame.draw.rect(self.window, (187, 255, 255),
                             (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                                     WINDOW_HEIGHT -  img_drone.get_height()  - self.height*3, 150, 100))
            self.height += 1
            ft_heightNumber = pygame.font.Font("sources/STZHONGS.TTF", 20)
            text_heightNumber = ft_heightNumber.render(str(self.height) + "m", True, (0, 0, 0))
            self.window.blit(text_heightNumber, (100, 450))

            self.window.blit(img_drone, (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                                         WINDOW_HEIGHT  - img_drone.get_height()  - self.height*3))

            pygame.display.update()
            time.sleep(0.1)
            pygame.draw.rect(self.window, (255, 245, 238),
                             (100, 450, 60, 50))



    def Down(self, height):
        # img_downarrow = pygame.image.load("sources/downarrow.png")
        # img_downarrow = pygame.transform.rotozoom(img_downarrow, 0, 0.2)
        # img_downarrowWidth, img_downarrowHeight = img_downarrow.get_size()
        # self.window.blit(img_downarrow, (leftWidth + middleWidth / 2 - img_downarrowWidth / 2, 500))
        # pygame.display.update()
        # for i in range(height):
        #     self.height -= 1
        #     ft_heightNumber = pygame.font.Font("sources/STZHONGS.TTF", 25)
        #     text_heightNumber = ft_heightNumber.render(str(self.height) + "m", True, (0, 0, 0))
        #     self.window.blit(text_heightNumber, (leftWidth + middleWidth / 2 + 20, 450))
        #     pygame.display.update()
        #     time.sleep(0.1)
        #     pygame.draw.rect(self.window, (187, 255, 255), (leftWidth + middleWidth / 2 + 10, 450, 80, 50))
        #
        img_drone = pygame.image.load("sources/drone.png")
        img_drone = pygame.transform.rotozoom(img_drone, 0, 0.25)  # 保持当前缩放比例
        # 调整图像大小
        self.window.blit(img_drone, (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                                     WINDOW_HEIGHT - img_drone.get_height() - self.height*3))
        for i in range(height):
            pygame.draw.rect(self.window, (187, 255, 255),
                             (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                              WINDOW_HEIGHT  - img_drone.get_height()  - self.height*3, 150, 100))
            self.height -= 1
            ft_heightNumber = pygame.font.Font("sources/STZHONGS.TTF", 20)
            text_heightNumber = ft_heightNumber.render(str(self.height) + "m", True, (0, 0, 0))
            self.window.blit(text_heightNumber, (100, 450))

            self.window.blit(img_drone, (leftWidth + middleWidth / 2 - img_drone.get_width() / 2,
                                         WINDOW_HEIGHT  - img_drone.get_height()  - self.height*3))

            pygame.display.update()
            time.sleep(0.1)
            pygame.draw.rect(self.window, (255, 245, 238),
                             (100, 450, 60, 50))

    def turnLeft(self):
        img_droneLVL = pygame.image.load("sources/droneLVL.png")
        img_droneLVL = pygame.transform.rotozoom(img_droneLVL, 0, 0.5)
        img_droneLVL = pygame.transform.rotate(img_droneLVL, self.angle)
        self.window.blit(img_droneLVL, img_droneLVL.get_rect(center=tuple(self.core)))
        ft_angleNumber = pygame.font.Font("sources/STZHONGS.TTF", 25)
        text_angleNumber = ft_angleNumber.render(str(self.angle) + "°", True, (0, 0, 0))
        pygame.display.update()
        for i in range(45):
            img_droneLVL = pygame.image.load("sources/droneLVL.png")
            img_droneLVL = pygame.transform.rotozoom(img_droneLVL, 0, 0.5)
            self.angle += 1
            img_droneLVL = pygame.transform.rotate(img_droneLVL, self.angle)
            text_angleNumber = ft_angleNumber.render(str(self.angle) + "°", True, (0, 0, 0))
            self.window.blit(img_droneLVL, img_droneLVL.get_rect(center=tuple(self.core)))
            self.window.blit(text_angleNumber, (leftWidth + middleWidth + rightWidth / 2 + 20, 600))
            pygame.display.update()
            time.sleep(0.067)
            pygame.draw.rect(self.window, (152, 245, 255),
                             (leftWidth + middleWidth + rightWidth / 2 + 10, 600, 80, 50))

    def turnRight(self):
        img_droneLVL = pygame.image.load("sources/droneLVL.png")
        img_droneLVL = pygame.transform.rotozoom(img_droneLVL, 0, 0.5)
        img_droneLVL = pygame.transform.rotate(img_droneLVL, self.angle)
        self.window.blit(img_droneLVL, img_droneLVL.get_rect(center=tuple(self.core)))
        ft_angleNumber = pygame.font.Font("sources/STZHONGS.TTF", 25)
        text_angleNumber = ft_angleNumber.render(str(self.angle) + "°", True, (0, 0, 0))
        pygame.display.update()
        for i in range(45):
            img_droneLVL = pygame.image.load("sources/droneLVL.png")
            img_droneLVL = pygame.transform.rotozoom(img_droneLVL, 0, 0.5)
            self.angle -= 1
            img_droneLVL = pygame.transform.rotate(img_droneLVL, self.angle)
            text_angleNumber = ft_angleNumber.render(str(self.angle) + "°", True, (0, 0, 0))
            self.window.blit(img_droneLVL, img_droneLVL.get_rect(center=tuple(self.core)))
            self.window.blit(text_angleNumber, (leftWidth + middleWidth + rightWidth / 2 + 20, 600))
            pygame.display.update()
            time.sleep(0.067)
            pygame.draw.rect(self.window, (152, 245, 255),
                             (leftWidth + middleWidth + rightWidth / 2 + 10, 600, 80, 50))

    def Quit(self):
        pygame.quit()
