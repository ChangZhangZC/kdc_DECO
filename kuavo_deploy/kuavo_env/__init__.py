from gymnasium.envs.registration import register
from kuavo_deploy.kuavo_env.KuavoSimEnv import KuavoSimEnv
from kuavo_deploy.kuavo_env.KuavoRealEnv import KuavoRealEnv
from kuavo_deploy.kuavo_env.KuavoDECOEnv import KuavoDECOEnv
register(
    id='Kuavo-Sim',
    entry_point='kuavo_deploy.kuavo_env.KuavoSimEnv:KuavoSimEnv',
)

register(
    id='Kuavo-Real',
    entry_point='kuavo_deploy.kuavo_env.KuavoRealEnv:KuavoRealEnv',
)

register(
    id='Kuavo-DECO',
    entry_point='kuavo_deploy.kuavo_env.KuavoDECOEnv:KuavoDECOEnv',
)
__all__ = ["KuavoSimEnv","KuavoRealEnv","KuavoDECOEnv"]