import re
from collections import deque

import numpy as np
import pika
import pika.exceptions

import config as c
from PIL import Image
from io import BytesIO
import base64


class TrainingQueue:
    connection: pika.BlockingConnection

    def __init__(self):
        self.training_queue = deque()
        self.connection = None
        self.channel = None

    def _data_callback(self, ch, methods, props, body):
        try:
            body = body.decode()
            which, left, right = body.split("||")
            left, right = re.sub('^data:image/.+;base64,', '', left), re.sub('^data:image/.+;base64,', '', right)
            left, right = Image.open(BytesIO(base64.b64decode(left))), Image.open(BytesIO(base64.b64decode(right)))
            self.training_queue.append((1 - 2 * int(which), left))
            self.training_queue.append((2 * int(which) - 1, right))
        except AttributeError as e:
            print(e)

        while len(self.training_queue) > c.n_samples_in_training:
            self.training_queue.popleft()

        ch.basic_ack(delivery_tag=methods.delivery_tag)

    def _set_up_connection(self):
        try:
            credentials = pika.PlainCredentials(username=c.pika_username, password=c.pika_passwd)
            parameters = pika.ConnectionParameters(host=c.pika_host, port=c.pika_port, credentials=credentials)
            self.connection = pika.BlockingConnection(parameters=parameters)
            self.channel = self.connection.channel()
            self.connection.process_data_events()
            self.channel.basic_consume(queue='NDS_up', on_message_callback=self._data_callback)
        except pika.exceptions.AMQPError as e:
            self.connection, self.channel = None, None

    def _try_ensuring_a_connection(self):
        if self.connection is None or self.connection.is_closed:
            self._set_up_connection()
        if self.connection is not None:
            try:
                self.connection.process_data_events()
            except pika.exceptions.AMQPError as e:
                self._set_up_connection()

    def push_image_pair(self, left, right):
        left = Image.fromarray(np.uint8(left * 255))
        right = Image.fromarray(np.uint8(right * 255))

        left_buffer = BytesIO()
        right_buffer = BytesIO()

        left.save(left_buffer, format='PNG')
        right.save(right_buffer, format='PNG')

        self._try_ensuring_a_connection()
        left = b'data:image/png;base64,' + base64.b64encode(left_buffer.getvalue())
        right = b'data:image/png;base64,' + base64.b64encode(right_buffer.getvalue())
        if self.channel is not None:
            self.channel.basic_publish(exchange="", routing_key="NDS_down",
                                       body="{}||{}".format(left.decode(), right.decode()))

    def await_data(self, time_limit):
        self._try_ensuring_a_connection()
        if self.connection is not None:
            self.connection.process_data_events(time_limit=time_limit)

    def queue_to_training_data(self):

        values = np.array(list(left for (left, _) in self.training_queue))
        bases64 = list(right for (_, right) in self.training_queue)
        images = np.array(list(np.array(image64) for image64 in bases64))/255
        values.shape = (*values.shape, 1)
        return images, values
