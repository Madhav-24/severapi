# Runtime configuration

Copy `.env.example` to `.env` on the Linux host. Set a generated API key and
the UUID of the production GPU (GPU 0). Keep `.env` out of source control and
restrict it to the service administrator (`chmod 600 .env`). Do not place API
keys in Compose files, shell history, or logs.

The API binds to loopback by default. To permit VPN clients, set
`INTERNAL_BIND_ADDRESS` to the host's approved VPN/internal interface address
and restrict TCP port `HOST_PORT` to authorised VPN subnets in the host and
network firewalls. Do not bind to `0.0.0.0` unless a documented firewall policy
permits it.
