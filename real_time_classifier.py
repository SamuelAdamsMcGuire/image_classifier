'''
script to use webcam and a transfer deep learning model to recognize
apples, oranges and bananas
'''
import cv2
import logging
from process import process
from datetime import datetime
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model


def get_args():
    parser = argparse.ArgumentParser(
        description='''
            takes pictures from the webcam. use <q> to end the session
            and <spacebar> to capture an image.
        '''
    )
    parser.add_argument(
        "name", help="folder name to store images", default='out')
    return parser.parse_args()


def key_action():
    # https://www.ascii-code.com/
    k = cv2.waitKey(10)
    if k == 113:  # q button
        return 'q'
    if k == 32:  # space bar
        return 'space'
    return None


def write_image(folder, frame, text):
    '''
    made a small tweak here that when the photo is save it also
    has the predicted category in the name
    '''
    time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out = f'{folder}/{time}.{text}.png'
    cv2.imwrite(out, frame)
    logging.info(f'written {out}')


def init_cam():
    logging.info('start web cam')
    cap = cv2.VideoCapture(0)

    # Check success
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set properties. Each returns === True on success (i.e. correct resolution)
    assert cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    assert cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    return cap


def add_text(text, frame):
    # Put some rectangular box on the image
    frame[-70:, :, :] = (191, 191, 191)
    cv2.putText(frame, text,
                org=(30, 445),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 0),
                thickness=1, lineType=2)


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.INFO)

    args = get_args()

    logging.info('create folder if not exists')
    Path(args.name).mkdir(parents=True, exist_ok=True)

    cap = init_cam()

    # initialize count to one
    count = 1

    # load transfered learning model
    model = load_model('models/a_b_o_2905.h5')

    key = None

    try:
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = cap.read()

            # fliping the image
            frame = cv2.flip(frame, 1)

            # stepping down by one
            count -= 1

            key = key_action()

            if key is None:
                # no key was pressed, count at 0 does a prediction every 32 counts
                if count == 0:
                    frame_pred = process(frame)
                    pred = model.predict(frame_pred)
                    logging.info(pred[0])
                    count = 32
                # simple prediction returns so that they can be used in the saving of the image when a picture is written
                if pred[0][0] > (pred[0][1] and pred[0][2]):
                    text = 'APPLE'
                    add_text(text, frame)
                elif pred[0][1] > (pred[0][0] and pred[0][2]):
                    text = 'BANANA'
                    add_text(text, frame)
                else:
                    text = 'ORANGE'
                    add_text(text, frame)

            if key == 'space':
                # space key was pressed, write the image without overlay
                write_image(args.name, frame, text)

            # Display the resulting frame
            cv2.imshow('frame', frame)

    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        cap.release()
        cv2.destroyAllWindows()
