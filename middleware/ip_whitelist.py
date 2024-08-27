from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, allowlist=None, allow_all=False):
        super().__init__(app)
        self.allowlist = allowlist or []
        self.allow_all = allow_all

    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host
        if self.allow_all or client_host in self.allowlist:
            response = await call_next(request)
        else:
            raise HTTPException(status_code=403, detail="Access forbidden")
        return response
