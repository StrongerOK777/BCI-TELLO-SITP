import sys
import tty
import termios

def init():
    """初始化终端模式（无需窗口）"""
    pass

def getKey(keyName):
    """
    从终端读取按键
    keyName: 按键名称 (如 'space', 'j', 'l', 'i', 'k')
    返回: True 如果按下了对应按键，False 否则
    """
    # 按键映射
    key_map = {
        'space': ' ',
        'j': 'j',
        'l': 'l',
        'i': 'i',
        'k': 'k',
    }
    
    # 获取要检测的按键对应的字符
    target_key = key_map.get(keyName.lower(), keyName)
    
    # 设置终端为非阻塞模式
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setraw(fd)
        # 尝试读取一个字符（非阻塞）
        ch = sys.stdin.read(1) if sys.stdin.readable() else ''
        
        # 检测 Ctrl+C (字符码为 \x03)
        if ch == '\x03':
            raise KeyboardInterrupt
        
        if ch == target_key:
            return True
    except KeyboardInterrupt:
        # 恢复终端设置后重新抛出异常
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        raise
    except:
        pass
    finally:
        # 恢复终端设置
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except:
            pass
    
    return False

if __name__ == '__main__':
    init()