""""
Application Entry Point
"""

import threading
import time
import webbrowser
import logging

logging.basicCongif(level=logging.INFO)
logger =logging.getLogger(__name__)

def main():
    """Launch the Malex assistant with a web UI"""
    host = "127.0.0.1"
    port = 8000

    config = uvicorn.Config("malex.server.app",host=host,port=port,log_level="info",reload=False)
    server =uvicorn.Server(config)

    thread = threading.Thread(target=server.run , daemon = True)
    thread.start()

    time.sleep(5)

    url =f"http://{host}/{port}"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)

    try:
        logger.info(" Malex started")
        while True:
            time.sleep(1)
    except:
        logger.info("\nMalex shutting down")
        server.should_exit = True

if __name__ = "__main__":
    main()
        