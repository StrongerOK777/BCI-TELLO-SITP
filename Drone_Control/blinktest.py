import os
import time
from neuropy import NeuroSkyPy


def main():
    port = os.getenv("MINDWAVE_PORT", "/dev/cu.usbmodem2017_2_251")
    baud = int(os.getenv("MINDWAVE_BAUD", "57600"))
    neuropy = NeuroSkyPy(port, baud)

    try:
        neuropy.start()
        while True:
            attention = neuropy.attention
            meditation = neuropy.meditation
            blink_strength = neuropy.blinkStrength
            poor_signal = neuropy.poorSignal
            delta = neuropy.delta
            theta = neuropy.theta
            low_alpha = neuropy.lowAlpha
            high_alpha = neuropy.highAlpha
            low_beta = neuropy.lowBeta
            high_beta = neuropy.highBeta
            low_gamma = neuropy.lowGamma
            mid_gamma = neuropy.midGamma
            print(
                "attention={attention} meditation={meditation} blinkStrength={blink} "
                "poorSignal={poor} delta={delta} theta={theta} lowAlpha={la} highAlpha={ha} "
                "lowBeta={lb} highBeta={hb} lowGamma={lg} midGamma={mg}".format(
                    attention=attention,
                    meditation=meditation,
                    blink=blink_strength,
                    poor=poor_signal,
                    delta=delta,
                    theta=theta,
                    la=low_alpha,
                    ha=high_alpha,
                    lb=low_beta,
                    hb=high_beta,
                    lg=low_gamma,
                    mg=mid_gamma,
                )
               
            )
            print("")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        neuropy.stop()


if __name__ == "__main__":
    main()
