import logging
import datetime
import os
import io


class CustomBaseRotationHandler(logging.FileHandler):
    namer = None
    rotator = None

    def __init__(self, filename, mode, encoding=None, delay=False, errors=None):
        """
        Use the specified filename for streamed logging
        """
        logging.FileHandler.__init__(self, filename, mode=mode,
                                     encoding=encoding, delay=delay,
                                     errors=errors)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors

    def emit(self, record):
        """
        Emit a record.

        Output the record to the file, catering for rollover as described
        in doRollover().
        """
        try:
            if self.shouldRollover(record):
                self.doRollover()
            logging.FileHandler.terminator = ',\n'
            logging.FileHandler.emit(self, record)
        except Exception:
            self.handleError(record)

    def rotate(self, source, dest):
        """
        When rotating, rotate the current log.

        The default implementation calls the 'rotator' attribute of the
        handler, if it's callable, passing the source and dest arguments to
        it. If the attribute isn't callable (the default is None), the source
        is simply renamed to the destination.

        :param source: The source filename. This is normally the base
                       filename, e.g. 'test.log'
        :param dest:   The destination filename. This is normally
                       what the source is rotated to, e.g. 'test.log.1'.
        """
        if not callable(self.rotator):
            # Issue 18940: A file may not have been created if delay is True.
            if os.path.exists(source):
                os.open(dest, os.O_APPEND | os.O_CREAT)
        else:
            self.rotator(source, dest)


class DailyRotatingFileHandler(CustomBaseRotationHandler):
    """
    Handler for logging to a set of files, which switches from one file
    to the next when the current file reaches a certain size.
    """
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0,
                 encoding=None, delay=False, errors=None):
        """
        Open the specified file and use it as the stream for logging.

        By default, the file grows indefinitely. You can specify particular
        values of maxBytes and backupCount to allow the file to rollover at
        a predetermined size.

        Rollover occurs whenever the current log file is nearly maxBytes in
        length. If backupCount is >= 1, the system will successively create
        new files with the same pathname as the base file, but with extensions
        ".1", ".2" etc. appended to it. For example, with a backupCount of 5
        and a base file name of "app.log", you would get "app.log",
        "app.log.1", "app.log.2", ... through to "app.log.5". The file being
        written to is always "app.log" - when it gets filled up, it is closed
        and renamed to "app.log.1", and if files "app.log.1", "app.log.2" etc.
        exist, then they are renamed to "app.log.2", "app.log.3" etc.
        respectively.

        If maxBytes is zero, rollover never occurs.
        """
        # If rotation/rollover is wanted, it doesn't make sense to use another
        # mode. If for example 'w' were specified, then if there were multiple
        # runs of the calling application, the logs from previous runs would be
        # lost if the 'w' is respected, because the log file would be truncated
        # on each run.
        if maxBytes > 0:
            mode = 'a'
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        CustomBaseRotationHandler.__init__(self, filename, mode, encoding=encoding,
                                     delay=delay, errors=errors)
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.original_filename = filename

    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        """
        if self.backupCount > 0:
            self.baseFilename = self.rotation_filename(self.get_base_filename())
        if not self.delay:
            self.stream = self._open()
            self.stream.write('[')

    def rotation_filename(self, default_name):
        """
        Modify the filename of a log file when rotating.

        This is provided so that a custom filename can be provided.

        The default implementation calls the 'namer' attribute of the
        handler, if it's callable, passing the default name to
        it. If the attribute isn't callable (the default is None), the name
        is returned unchanged.

        :param default_name: The default name for the log file.
        """
        if not callable(self.namer):
            result = default_name
        else:
            result = self.namer(default_name)
        return result

    def get_base_filename(self):
        """
        @summary: Return logFile name string formatted to "Facelogs.today.json"
        """
        if self.baseFilename.split("\\")[-1] != self.original_filename.split("/")[-1]:
            token_one = "Part_"
            token_two = ".json"
            current_part = ''.join(self.baseFilename.split(token_one)[1].split(token_two)[0])
            if current_part is not None:
                next_part = int(current_part) + 1
        else:
            next_part = 1
        basename_ = self.original_filename + '_' + datetime.date.today().strftime("%Y-%m-%d") + '_Part_' + str(next_part) + ".json"
        return basename_


    def shouldRollover(self, record):
        """
        @summary:
        Rollover happen
        1. When the logFile size is get over maxBytes.
        2. When date is changed.
        @see: BaseRotatingHandler.emit
        """

        if self.stream is None:
            self.stream = self._open()

        if int(self.maxBytes) > 0:
            msg = "%s\n" % self.format(record)
            self.stream.seek(0, 2)
            if self.stream.tell() + len(msg) >= int(self.maxBytes):
                self.stream.write(']')
                return 1

        if datetime.date.today().strftime("%Y-%m-%d") not in self.baseFilename.split("/")[-1]:
            return 1
        return 0
