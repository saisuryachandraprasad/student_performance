from src.logger import logging
import sys

""" format of error message"""
def error_message_detail(error, error_detail:sys):
    _,_,_exc_tb = error_detail.exc_info()
    filename = _exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script [{0}] line number [{1}] and message is [{2}]".format(
        filename,_exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self. error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
    

if __name__ =="__main__":
    logging.info('exception test')