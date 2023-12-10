from contextlib import contextmanager
import requests

class API:
    def __init__(self, domain):
        self.domain = domain

    @classmethod
    @contextmanager
    def error_handling(cls):
        try:
            yield
        except Exception as e:
            print('[API 요청 실패]')
            print(e)
            raise e        
    
    def get(self, path: str, params: dict={}, headers: dict={}
            ) -> requests.Response:
        with self.error_handling():
            response = requests.post(
                url=f'{self.domain}/{path}',
                params=params,
                headers=headers, 
                timeout=2000)
            response.raise_for_status()
            return response
        
    def post(self, path: str, payload: dict
             ) -> requests.Response:
        with self.error_handling():
            response = requests.post(
                url=f'{self.domain}/{path}',
                json=payload, 
                timeout=2000)
            status_code = response.status_code
            response.raise_for_status()
            return response