import sys
import platform

# 跨平台键盘输入支持
if platform.system() == 'Windows':
    import msvcrt
    
    # 用于存储上一次读取的字符
    _last_char = None
    _char_consumed = False
    
    def init():
        """初始化（Windows 版本）"""
        global _last_char, _char_consumed
        _last_char = None
        _char_consumed = False
    
    def getKey(keyName):
        """
        Windows 版本：检测按键（和 test.py 兼容）
        keyName: 按键名称 (如 'SPACE', 'j', 'l', 'i', 'k')
        返回: True 如果按下了对应按键，False 否则
        """
        global _last_char, _char_consumed
        
        key_map = {
            'SPACE': ' ',
            'J': 'j',
            'L': 'l',
            'I': 'i',
            'K': 'k',
        }
        
        target_key = key_map.get(keyName.upper(), keyName.lower())
        
        # 如果上一个字符还没有被消耗，继续检查它
        if _last_char is not None and not _char_consumed:
            ch = _last_char
            if ch == target_key:
                _char_consumed = True
                return True
            # 如果按键被消耗，继续读取新的
            if _char_consumed:
                _last_char = None
        
        # 读取新的字符
        if not msvcrt.kbhit():
            return False
        
        try:
            ch = msvcrt.getch()
            if isinstance(ch, bytes):
                ch = ch.decode('utf-8', errors='ignore')
            ch = ch.lower()
            
            # 保存这个字符供后续检查
            _last_char = ch
            _char_consumed = False
            
            # 检查是否匹配
            if ch == target_key:
                _char_consumed = True
                return True
            
            return False
        except:
            return False

else:
    # Unix/Linux 版本
    import tty
    import termios
    
    def init():
        """初始化终端模式（Unix/Linux 版本）"""
        pass
    
    def getKey(keyName):
        """
        Unix/Linux 版本：从终端读取按键
        keyName: 按键名称
        返回: True 如果按下了对应按键，False 否则
        """
        key_map = {
            'SPACE': ' ',
            'j': 'j',
            'l': 'l',
            'i': 'i',
            'k': 'k',
        }
        
        target_key = key_map.get(keyName.upper(), keyName.lower())
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1) if sys.stdin.readable() else ''
            
            if ch == '\x03':
                raise KeyboardInterrupt
            
            if ch == target_key:
                return True
        except KeyboardInterrupt:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            raise
        except:
            pass
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except:
                pass
        
        return False

if __name__ == '__main__':
    init()