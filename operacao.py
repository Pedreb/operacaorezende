import streamlit as st
import pandas as pd
import sqlite3
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import base64
import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_base64_image(image_path):
    """Converte imagem local para base64 para exibição no Streamlit"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


# Configuração da página
st.set_page_config(
    page_title="BI Rezende Energia",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# CSS Personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .custom-header {
        background: linear-gradient(135deg, #F7931E 0%, #e67e22 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(247, 147, 30, 0.3);
        color: white;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
    }

    .custom-header .logo-container {
        flex-shrink: 0;
    }

    .custom-header .logo-container img {
        height: 80px;
        width: auto;
        filter: brightness(0) invert(1);
    }

    .custom-header .text-container {
        flex-grow: 1;
    }

    .custom-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .custom-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #F7931E;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .cache-status {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 1rem 0;
        font-size: 0.85rem;
        color: #2e7d32;
    }

    .cache-status.updating {
        background: #fff3e0;
        border-color: #ff9800;
        color: #f57c00;
    }

    .cache-status.error {
        background: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }

    .stButton > button {
        background: linear-gradient(135deg, #F7931E 0%, #e67e22 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(247, 147, 30, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(247, 147, 30, 0.4);
    }

    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #F7931E;
    }

    [data-testid="metric-container"] > div {
        color: #000000;
    }

    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
    }

    .chart-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
    }

    @media (max-width: 768px) {
        .custom-header {
            flex-direction: column;
            gap: 1rem;
        }
        .custom-header h1 {
            font-size: 2rem;
        }
        .custom-header .logo-container img {
            height: 60px;
        }
    }
</style>
""", unsafe_allow_html=True)


class DataManager:
    """Gerenciador de dados com modelo dimensional e cache SQLite"""

    def __init__(self):
        self.db_path = "rezende_cache.db"
        self.cache_duration_hours = 1  # Cache válido por 1 hora
        self.daily_cache_duration = 24  # Cache diário válido por 24 horas

        # Verificar se estamos no Streamlit Cloud
        self.is_cloud = os.environ.get('STREAMLIT_CLOUD', False)

        # Equipes excluídas do faturamento
        self.equipes_excluidas = [
            'RZ_JUPITER_01',
            'RZ_JUPITER_02',
            'RZ_CONCRETO_01',
            'EQ_TREINAMENTO_NW'
        ]
        try:
            self.redshift_config = {
                "host": st.secrets["redshift"]["host"],
                "port": st.secrets["redshift"]["port"],
                "database": st.secrets["redshift"]["database"],
                "user": st.secrets["redshift"]["user"],
                "password": st.secrets["redshift"]["password"]
            }
        except Exception as e:
            logger.error(f"Erro ao carregar configurações do Redshift: {e}")
            raise

            # INICIALIZAR o cache automaticamente
        try:
            self.init_cache_db()
        except Exception as e:
            logger.error(f"Erro ao inicializar cache: {e}")

    def get_equipes_excluidas_sql(self):
        """Retorna cláusula SQL para excluir equipes específicas"""
        equipes_str = "', '".join(self.equipes_excluidas)
        return f"AND s.des_equipe_rcb NOT IN ('{equipes_str}')"

    def health_check(self):
        """Verifica se o sistema está funcionando corretamente"""
        checks = {
            'sqlite_connection': False,
            'redshift_connection': False,
            'cache_initialized': False,
            'tables_exist': False
        }

        try:
            # Teste SQLite
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
                checks['sqlite_connection'] = True

                # Verificar se tabelas existem
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name IN ('cache_metadata', 'dim_calendario', 'dim_equipes', 'fato_servicos')
                """)
                table_count = cursor.fetchone()[0]
                checks['tables_exist'] = table_count >= 4
                checks['cache_initialized'] = table_count > 0

        except Exception as e:
            logger.error(f"Erro no health check SQLite: {e}")

        try:
            # Teste Redshift
            with self.get_redshift_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    checks['redshift_connection'] = True
        except Exception as e:
            logger.error(f"Erro no health check Redshift: {e}")

        return checks

    def get_equipes_excluidas_sqlite(self):
        """Retorna cláusula SQLite para excluir equipes específicas (por nome da equipe)"""
        placeholders = ','.join(['?' for _ in self.equipes_excluidas])
        return f"AND nome_equipe NOT IN ({placeholders})", self.equipes_excluidas

    def init_cache_db(self):
        """Inicializa o banco SQLite de cache com modelo dimensional"""
        with sqlite3.connect(self.db_path) as conn:
            # Tabela de metadados do cache
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    table_name TEXT PRIMARY KEY,
                    last_update TIMESTAMP,
                    record_count INTEGER,
                    status TEXT DEFAULT 'active'
                )
            """)

            # DIMENSÃO CALENDÁRIO
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dim_calendario (
                    data_key DATE PRIMARY KEY,
                    ano INTEGER,
                    mes INTEGER,
                    dia INTEGER,
                    dia_semana INTEGER,
                    nome_dia_semana TEXT,
                    nome_mes TEXT,
                    trimestre INTEGER,
                    semana_ano INTEGER,
                    is_fim_semana BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # DIMENSÃO EQUIPES
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dim_equipes (
                    cod_equipe TEXT PRIMARY KEY,
                    nome_equipe TEXT,
                    regional TEXT,
                    is_ativa BOOLEAN DEFAULT TRUE,
                    is_excluida BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # FATO SERVIÇOS
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fato_servicos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_execucao DATE,
                    cod_equipe TEXT,
                    nome_equipe TEXT,
                    descricao_servico TEXT,
                    quantidade INTEGER,
                    valor_total REAL,
                    valor_unitario REAL,
                    regional TEXT,
                    nota TEXT,
                    placa TEXT,
                    cidade TEXT,
                    cod_obra TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_execucao) REFERENCES dim_calendario(data_key),
                    FOREIGN KEY (cod_equipe) REFERENCES dim_equipes(cod_equipe)
                )
            """)

            # FATO METAS
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fato_metas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_meta DATE,
                    cod_equipe TEXT,
                    nome_equipe TEXT,
                    valor_meta REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_meta) REFERENCES dim_calendario(data_key),
                    FOREIGN KEY (cod_equipe) REFERENCES dim_equipes(cod_equipe)
                )
            """)

            # FATO FATURAMENTO DIÁRIO
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fato_faturamento_diario (
                    data DATE PRIMARY KEY,
                    faturamento_diario REAL,
                    servicos_dia INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data) REFERENCES dim_calendario(data_key)
                )
            """)

            # Índices para performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fato_servicos_data ON fato_servicos(data_execucao)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fato_servicos_equipe ON fato_servicos(cod_equipe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fato_servicos_nome_equipe ON fato_servicos(nome_equipe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fato_metas_data ON fato_metas(data_meta)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fato_metas_equipe ON fato_metas(cod_equipe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_calendario_ano_mes ON dim_calendario(ano, mes)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dim_equipes_regional ON dim_equipes(regional)")

            conn.commit()

    @contextmanager
    def get_redshift_connection(self):
        """Context manager para conexões com Redshift"""
        conn = None
        try:
            conn = psycopg2.connect(**self.redshift_config)
            yield conn
        except Exception as e:
            logger.error(f"Erro na conexão Redshift: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def needs_refresh(self, table_name, custom_hours=None):
        """Verifica se o cache precisa ser atualizado"""
        hours = custom_hours or self.cache_duration_hours

        try:
            # Garantir que o banco está inicializado
            self.init_cache_db()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Verificar se a tabela cache_metadata existe
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='cache_metadata'
                """)

                if not cursor.fetchone():
                    return True, "Cache não inicializado"

                cursor.execute("""
                    SELECT last_update, record_count FROM cache_metadata 
                    WHERE table_name = ?
                """, (table_name,))

                result = cursor.fetchone()
                if not result or not result[0]:
                    return True, "Cache não existe"

                last_update = datetime.fromisoformat(result[0])
                time_diff = datetime.now() - last_update

                if time_diff > timedelta(hours=hours):
                    return True, f"Cache expirado ({time_diff})"

                return False, f"Cache válido (atualizado há {time_diff})"

        except Exception as e:
            logger.error(f"Erro ao verificar cache {table_name}: {e}")
            return True, f"Erro na verificação: {e}"

    def update_cache_metadata(self, table_name, record_count):
        """Atualiza metadados do cache"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache_metadata 
                (table_name, last_update, record_count, status)
                VALUES (?, ?, ?, 'active')
            """, (table_name, datetime.now().isoformat(), record_count))
            conn.commit()

    def refresh_dimensao_calendario(self, force=False):
        """Atualiza dimensão calendário"""
        needs_update, reason = self.needs_refresh('dim_calendario', 168)  # 7 dias

        if not needs_update and not force:
            return False, reason

        try:
            logger.info("Iniciando atualização da dimensão calendário...")

            # Gerar calendário para os últimos 2 anos e próximos 1 ano
            start_date = datetime.now() - timedelta(days=730)  # 2 anos atrás
            end_date = datetime.now() + timedelta(days=365)  # 1 ano à frente

            calendar_data = []
            current_date = start_date

            while current_date <= end_date:
                calendar_data.append({
                    'data_key': current_date.strftime('%Y-%m-%d'),
                    'ano': current_date.year,
                    'mes': current_date.month,
                    'dia': current_date.day,
                    'dia_semana': current_date.weekday() + 1,  # 1=Segunda, 7=Domingo
                    'nome_dia_semana': current_date.strftime('%A'),
                    'nome_mes': current_date.strftime('%B'),
                    'trimestre': (current_date.month - 1) // 3 + 1,
                    'semana_ano': current_date.isocalendar()[1],
                    'is_fim_semana': current_date.weekday() >= 5  # Sábado=5, Domingo=6
                })
                current_date += timedelta(days=1)

            df_calendario = pd.DataFrame(calendar_data)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM dim_calendario")
                df_calendario.to_sql('dim_calendario', conn, if_exists='append', index=False)
                conn.commit()

            self.update_cache_metadata('dim_calendario', len(df_calendario))
            logger.info(f"Dimensão calendário atualizada: {len(df_calendario)} registros")

            return True, f"Dimensão calendário atualizada com {len(df_calendario)} registros"

        except Exception as e:
            logger.error(f"Erro ao atualizar dimensão calendário: {e}")
            return False, f"Erro: {e}"

    def refresh_dimensao_equipes(self, force=False):
        """Atualiza dimensão equipes"""
        needs_update, reason = self.needs_refresh('dim_equipes')

        if not needs_update and not force:
            return False, reason

        try:
            logger.info("Iniciando atualização da dimensão equipes...")

            # Query para buscar equipes únicas do Redshift
            query = """
            SELECT DISTINCT 
                s.cod_equipe_rcb as cod_equipe,
                s.des_equipe_rcb as nome_equipe,
                s.num_cont_rcb as regional
            FROM res_cubo_servico s
            WHERE s.cod_equipe_rcb IS NOT NULL 
                AND s.des_equipe_rcb IS NOT NULL
            UNION
            SELECT DISTINCT 
                m.cod_equipe_cms as cod_equipe,
                s.des_equipe_rcb as nome_equipe,
                s.num_cont_rcb as regional
            FROM res_cubo_servico_meta m
            LEFT JOIN res_cubo_servico s ON m.cod_equipe_cms = s.cod_equipe_rcb
            WHERE m.cod_equipe_cms IS NOT NULL
            ORDER BY cod_equipe
            """

            with self.get_redshift_connection() as conn:
                df_equipes = pd.read_sql_query(query, conn)

            if df_equipes.empty:
                return False, "Nenhuma equipe encontrada no Redshift"

            # Marcar equipes excluídas
            df_equipes['is_excluida'] = df_equipes['nome_equipe'].isin(self.equipes_excluidas)
            df_equipes['is_ativa'] = ~df_equipes['is_excluida']

            # Limpar dados anteriores e inserir novos
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM dim_equipes")
                df_equipes.to_sql('dim_equipes', conn, if_exists='append', index=False)
                conn.commit()

            self.update_cache_metadata('dim_equipes', len(df_equipes))
            logger.info(f"Dimensão equipes atualizada: {len(df_equipes)} registros")

            return True, f"Dimensão equipes atualizada com {len(df_equipes)} registros"

        except Exception as e:
            logger.error(f"Erro ao atualizar dimensão equipes: {e}")
            return False, f"Erro: {e}"

    def refresh_fato_servicos(self, force=False):
        """Atualiza fato serviços"""
        needs_update, reason = self.needs_refresh('fato_servicos')

        if not needs_update and not force:
            return False, reason

        try:
            logger.info("Iniciando atualização do fato serviços...")

            # Query principal do Redshift
            query = f"""
            SELECT 
                s.dta_exec_rcb as data_execucao,
                s.cod_equipe_rcb as cod_equipe,
                s.des_equipe_rcb as nome_equipe,
                s.des_preco_pre_rcb as descricao_servico,
                COUNT(*) as quantidade,
                COALESCE(SUM(s.vlr_total_serv_rcb), 0) as valor_total,
                CASE 
                    WHEN COUNT(*) > 0 THEN COALESCE(SUM(s.vlr_total_serv_rcb), 0) / COUNT(*)
                    ELSE 0 
                END as valor_unitario,
                s.num_cont_rcb as regional,
                COALESCE(o.cod_pep, 'N/A') as nota,
                s.cod_placa_rcb as placa,
                COALESCE(s.des_cida_rcb, 'N/A') as cidade,
                s.cod_obra_rcb as cod_obra
            FROM res_cubo_servico s
            LEFT JOIN res_cubo_obra o ON s.cod_obra_rcb = o.cod_obra
            WHERE s.dta_exec_rcb IS NOT NULL 
                AND s.dta_exec_rcb >= CURRENT_DATE - INTERVAL '90 days'
                {self.get_equipes_excluidas_sql()}
            GROUP BY 
                s.dta_exec_rcb, s.cod_equipe_rcb, s.des_equipe_rcb, s.des_preco_pre_rcb, s.num_cont_rcb, 
                o.cod_pep, s.cod_placa_rcb, s.des_cida_rcb, s.cod_obra_rcb
            ORDER BY s.dta_exec_rcb DESC
            """

            with self.get_redshift_connection() as conn:
                df_servicos = pd.read_sql_query(query, conn)

            if df_servicos.empty:
                return False, "Nenhum dado retornado do Redshift"

            # Limpar cache antigo e inserir novos dados
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM fato_servicos")
                df_servicos.to_sql('fato_servicos', conn, if_exists='append', index=False)
                conn.commit()

            self.update_cache_metadata('fato_servicos', len(df_servicos))
            logger.info(f"Fato serviços atualizado: {len(df_servicos)} registros")

            return True, f"Fato serviços atualizado com {len(df_servicos)} registros"

        except Exception as e:
            logger.error(f"Erro ao atualizar fato serviços: {e}")
            return False, f"Erro: {e}"

    def refresh_fato_metas(self, force=False):
        """Atualiza fato metas"""
        needs_update, reason = self.needs_refresh('fato_metas')

        if not needs_update and not force:
            return False, reason

        try:
            logger.info("Iniciando atualização do fato metas...")

            # Query para buscar dados de meta do Redshift
            query = f"""
            SELECT 
                m.dta_meta_msd as data_meta,
                m.cod_equipe_cms as cod_equipe,
                s.des_equipe_rcb as nome_equipe,
                COALESCE(m.vlr_meta_msd, 0) as valor_meta
            FROM res_cubo_servico_meta m
            LEFT JOIN res_cubo_servico s ON m.cod_equipe_cms = s.cod_equipe_rcb
            WHERE m.dta_meta_msd IS NOT NULL 
                AND m.dta_meta_msd >= CURRENT_DATE - INTERVAL '90 days'
                AND s.des_equipe_rcb IS NOT NULL
                AND s.des_equipe_rcb NOT IN ('{"', '".join(self.equipes_excluidas)}')
            GROUP BY m.dta_meta_msd, m.cod_equipe_cms, s.des_equipe_rcb, m.vlr_meta_msd
            ORDER BY m.dta_meta_msd DESC
            """

            with self.get_redshift_connection() as conn:
                df_metas = pd.read_sql_query(query, conn)

            if not df_metas.empty:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM fato_metas")
                    df_metas.to_sql('fato_metas', conn, if_exists='append', index=False)
                    conn.commit()

                self.update_cache_metadata('fato_metas', len(df_metas))
                logger.info(f"Fato metas atualizado: {len(df_metas)} registros")
                return True, f"Fato metas atualizado com {len(df_metas)} registros"

            return False, "Nenhum dado de meta encontrado"

        except Exception as e:
            logger.error(f"Erro ao atualizar fato metas: {e}")
            return False, f"Erro: {e}"

    def refresh_fato_faturamento_diario(self, force=False):
        """Atualiza fato faturamento diário"""
        needs_update, reason = self.needs_refresh('fato_faturamento_diario')

        if not needs_update and not force:
            return False, reason

        try:
            logger.info("Iniciando atualização do fato faturamento diário...")

            # Calcular a partir do fato_servicos (já filtrado)
            query = """
            SELECT 
                data_execucao as data,
                SUM(valor_total) as faturamento_diario,
                SUM(quantidade) as servicos_dia
            FROM fato_servicos
            GROUP BY data_execucao
            ORDER BY data_execucao
            """

            with sqlite3.connect(self.db_path) as conn:
                df_faturamento = pd.read_sql_query(query, conn)

            if not df_faturamento.empty:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM fato_faturamento_diario")
                    df_faturamento.to_sql('fato_faturamento_diario', conn, if_exists='append', index=False)
                    conn.commit()

                self.update_cache_metadata('fato_faturamento_diario', len(df_faturamento))
                logger.info(f"Fato faturamento diário atualizado: {len(df_faturamento)} registros")
                return True, f"Fato faturamento diário atualizado com {len(df_faturamento)} registros"

            return False, "Nenhum dado de faturamento encontrado"

        except Exception as e:
            logger.error(f"Erro ao atualizar fato faturamento diário: {e}")
            return False, f"Erro: {e}"

    def refresh_all_caches(self, force=False):
        """Atualiza todos os caches seguindo a ordem dimensional"""
        results = {}

        # 1. Dimensões primeiro
        success, message = self.refresh_dimensao_calendario(force)
        results['dim_calendario'] = {'success': success, 'message': message}

        success, message = self.refresh_dimensao_equipes(force)
        results['dim_equipes'] = {'success': success, 'message': message}

        # 2. Fatos depois
        success, message = self.refresh_fato_servicos(force)
        results['fato_servicos'] = {'success': success, 'message': message}

        success, message = self.refresh_fato_metas(force)
        results['fato_metas'] = {'success': success, 'message': message}

        success, message = self.refresh_fato_faturamento_diario(force)
        results['fato_faturamento_diario'] = {'success': success, 'message': message}

        return results

    def get_dimension_filters(self):
        """Busca opções de filtro das DIMENSÕES - CORRIGIDO"""
        options = {}

        with sqlite3.connect(self.db_path) as conn:
            # EQUIPES - buscar do FATO (dados reais) em vez da dimensão
            df = pd.read_sql_query("""
                SELECT DISTINCT 
                    fs.cod_equipe, 
                    fs.nome_equipe, 
                    fs.regional 
                FROM fato_servicos fs
                WHERE fs.nome_equipe IS NOT NULL
                ORDER BY fs.nome_equipe
            """, conn)
            options['equipes'] = df['nome_equipe'].tolist()
            options['cod_equipes'] = df['cod_equipe'].tolist()

            # REGIONAIS - buscar do FATO também
            df_regionais = pd.read_sql_query("""
                SELECT DISTINCT regional 
                FROM fato_servicos 
                WHERE regional IS NOT NULL
                ORDER BY regional
            """, conn)
            options['regionais'] = df_regionais['regional'].tolist()

            # Anos (da dimensão calendário - OK)
            df = pd.read_sql_query("""
                SELECT DISTINCT ano 
                FROM dim_calendario 
                ORDER BY ano DESC
            """, conn)
            options['anos'] = df['ano'].tolist()

            # Meses (OK)
            df = pd.read_sql_query("""
                SELECT DISTINCT mes, nome_mes 
                FROM dim_calendario 
                ORDER BY mes
            """, conn)
            options['meses'] = df['nome_mes'].tolist()

            # Tipos de serviço (do fato - OK)
            df = pd.read_sql_query("""
                SELECT DISTINCT descricao_servico 
                FROM fato_servicos 
                WHERE descricao_servico IS NOT NULL 
                ORDER BY descricao_servico
            """, conn)
            options['tipos_servico'] = df['descricao_servico'].tolist()

            # Notas (do fato - OK)
            df = pd.read_sql_query("""
                SELECT DISTINCT nota 
                FROM fato_servicos 
                WHERE nota IS NOT NULL AND nota != 'N/A'
                ORDER BY nota
            """, conn)
            options['notas'] = df['nota'].tolist()

            # Placas (do fato - OK)
            df = pd.read_sql_query("""
                SELECT DISTINCT placa 
                FROM fato_servicos 
                WHERE placa IS NOT NULL 
                ORDER BY placa
            """, conn)
            options['placas'] = df['placa'].tolist()

        return options
    def get_servicos_data(self, filters=None):
        """Busca dados de serviços com JOINs dimensionais"""
        query = """
        SELECT 
            fs.*,
            de.regional as regional_equipe,
            dc.ano,
            dc.mes,
            dc.nome_mes,
            dc.nome_dia_semana
        FROM fato_servicos fs
        LEFT JOIN dim_equipes de ON fs.cod_equipe = de.cod_equipe
        LEFT JOIN dim_calendario dc ON fs.data_execucao = dc.data_key
        WHERE de.is_excluida = FALSE OR de.is_excluida IS NULL
        """
        params = []

        if filters:
            if filters.get('data_inicio'):
                query += " AND fs.data_execucao >= ?"
                params.append(filters['data_inicio'])

            if filters.get('data_fim'):
                query += " AND fs.data_execucao <= ?"
                params.append(filters['data_fim'])

            if filters.get('equipes'):
                placeholders = ','.join(['?' for _ in filters['equipes']])
                query += f" AND fs.nome_equipe IN ({placeholders})"
                params.extend(filters['equipes'])

            if filters.get('tipos_servico'):
                placeholders = ','.join(['?' for _ in filters['tipos_servico']])
                query += f" AND fs.descricao_servico IN ({placeholders})"
                params.extend(filters['tipos_servico'])

            if filters.get('regionais'):
                placeholders = ','.join(['?' for _ in filters['regionais']])
                query += f" AND fs.regional IN ({placeholders})"
                params.extend(filters['regionais'])

            if filters.get('notas'):
                placeholders = ','.join(['?' for _ in filters['notas']])
                query += f" AND fs.nota IN ({placeholders})"
                params.extend(filters['notas'])

            if filters.get('placas'):
                placeholders = ','.join(['?' for _ in filters['placas']])
                query += f" AND fs.placa IN ({placeholders})"
                params.extend(filters['placas'])

        query += " ORDER BY fs.data_execucao DESC"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_producao_vs_meta_data(self, filters=None):
        """Busca dados de produção vs meta com JOINs dimensionais - Meta acompanha os filtros exatos"""

        # Query para produção (faturamento real) - aplicando TODOS os filtros
        query_producao = """
        SELECT 
            fs.nome_equipe as equipe,
            fs.cod_equipe,
            SUM(fs.valor_total) as producao_real,
            SUM(fs.quantidade) as quantidade_servicos
        FROM fato_servicos fs
        LEFT JOIN dim_equipes de ON fs.cod_equipe = de.cod_equipe
        LEFT JOIN dim_calendario dc ON fs.data_execucao = dc.data_key
        WHERE (de.is_excluida = FALSE OR de.is_excluida IS NULL)
        """
        params_producao = []

        if filters:
            if filters.get('data_inicio'):
                query_producao += " AND fs.data_execucao >= ?"
                params_producao.append(filters['data_inicio'])

            if filters.get('data_fim'):
                query_producao += " AND fs.data_execucao <= ?"
                params_producao.append(filters['data_fim'])

            if filters.get('equipes'):
                placeholders = ','.join(['?' for _ in filters['equipes']])
                query_producao += f" AND fs.nome_equipe IN ({placeholders})"
                params_producao.extend(filters['equipes'])

            if filters.get('tipos_servico'):
                placeholders = ','.join(['?' for _ in filters['tipos_servico']])
                query_producao += f" AND fs.descricao_servico IN ({placeholders})"
                params_producao.extend(filters['tipos_servico'])

            if filters.get('regionais'):
                placeholders = ','.join(['?' for _ in filters['regionais']])
                query_producao += f" AND fs.regional IN ({placeholders})"
                params_producao.extend(filters['regionais'])

            if filters.get('notas'):
                placeholders = ','.join(['?' for _ in filters['notas']])
                query_producao += f" AND fs.nota IN ({placeholders})"
                params_producao.extend(filters['notas'])

            if filters.get('placas'):
                placeholders = ','.join(['?' for _ in filters['placas']])
                query_producao += f" AND fs.placa IN ({placeholders})"
                params_producao.extend(filters['placas'])

        query_producao += """
        GROUP BY fs.nome_equipe, fs.cod_equipe
        ORDER BY producao_real DESC
        """

        # Query para metas - APLICANDO OS MESMOS FILTROS DE DATA E EQUIPE que a produção
        query_meta = """
        SELECT 
            fm.nome_equipe as equipe,
            fm.cod_equipe,
            SUM(fm.valor_meta) as meta_periodo
        FROM fato_metas fm
        LEFT JOIN dim_equipes de ON fm.cod_equipe = de.cod_equipe
        LEFT JOIN dim_calendario dc ON fm.data_meta = dc.data_key
        WHERE (de.is_excluida = FALSE OR de.is_excluida IS NULL)
            AND fm.valor_meta > 0
        """
        params_meta = []

        # APLICAR OS MESMOS FILTROS DE DATA da produção
        if filters:
            if filters.get('data_inicio'):
                query_meta += " AND fm.data_meta >= ?"
                params_meta.append(filters['data_inicio'])

            if filters.get('data_fim'):
                query_meta += " AND fm.data_meta <= ?"
                params_meta.append(filters['data_fim'])

            # APLICAR OS MESMOS FILTROS DE EQUIPE da produção
            if filters.get('equipes'):
                placeholders = ','.join(['?' for _ in filters['equipes']])
                query_meta += f" AND fm.nome_equipe IN ({placeholders})"
                params_meta.extend(filters['equipes'])

        query_meta += " GROUP BY fm.nome_equipe, fm.cod_equipe"

        with sqlite3.connect(self.db_path) as conn:
            # Buscar produção
            df_producao = pd.read_sql_query(query_producao, conn, params=params_producao)

            # Buscar metas com os MESMOS filtros
            df_metas = pd.read_sql_query(query_meta, conn, params=params_meta)

        # Se não há produção, retornar vazio
        if df_producao.empty:
            return pd.DataFrame()

        logger.info(f"Produção: {len(df_producao)} equipes")
        logger.info(f"Metas: {len(df_metas)} equipes no mesmo período")

        # As metas já estão filtradas pelo período correto, então usar diretamente
        if not df_metas.empty:
            df_metas['meta_equipe'] = df_metas['meta_periodo']
            periodo_info = ""
            if filters and filters.get('data_inicio') and filters.get('data_fim'):
                start_date = datetime.fromisoformat(filters['data_inicio'])
                end_date = datetime.fromisoformat(filters['data_fim'])
                dias_periodo = (end_date - start_date).days + 1
                periodo_info = f" ({dias_periodo} dias)"

            logger.info(f"Meta no período{periodo_info}: R$ {df_metas['meta_equipe'].sum():,.2f}")
        else:
            logger.warning("Nenhuma meta encontrada no período selecionado")

        # Fazer merge entre produção e metas (LEFT JOIN para manter todas as equipes com produção)
        if not df_metas.empty:
            df_resultado = df_producao.merge(
                df_metas[['equipe', 'meta_equipe']],
                on='equipe',
                how='left'
            )
            df_resultado['meta_equipe'] = df_resultado['meta_equipe'].fillna(0)
        else:
            df_resultado = df_producao.copy()
            df_resultado['meta_equipe'] = 0

        # Calcular percentual de atingimento
        df_resultado['percentual_meta'] = np.where(
            df_resultado['meta_equipe'] > 0,
            (df_resultado['producao_real'] / df_resultado['meta_equipe']) * 100,
            0
        )

        logger.info(f"Resultado final: {len(df_resultado)} equipes")
        if not df_resultado.empty:
            logger.info(f"Total faturamento: R$ {df_resultado['producao_real'].sum():,.2f}")
            logger.info(f"Total meta: R$ {df_resultado['meta_equipe'].sum():,.2f}")
            equipes_com_meta = len(df_resultado[df_resultado['meta_equipe'] > 0])
            logger.info(f"Equipes com meta no período: {equipes_com_meta}")

        return df_resultado

    def get_faturamento_data(self, filters=None):
        """Busca dados de faturamento diário com JOINs dimensionais"""
        query = """
        SELECT 
            ffd.*,
            dc.ano,
            dc.mes,
            dc.nome_mes,
            dc.nome_dia_semana,
            dc.is_fim_semana
        FROM fato_faturamento_diario ffd
        LEFT JOIN dim_calendario dc ON ffd.data = dc.data_key
        WHERE 1=1
        """
        params = []

        if filters:
            if filters.get('data_inicio'):
                query += " AND ffd.data >= ?"
                params.append(filters['data_inicio'])

            if filters.get('data_fim'):
                query += " AND ffd.data <= ?"
                params.append(filters['data_fim'])

        query += " ORDER BY ffd.data"

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_cache_status(self):
        """Retorna status dos caches com verificação de existência"""
        try:
            # Garantir que o banco está inicializado
            self.init_cache_db()

            with sqlite3.connect(self.db_path) as conn:
                # Verificar se a tabela existe
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='cache_metadata'
                """)

                if not cursor.fetchone():
                    # Se não existe, retornar DataFrame vazio
                    return pd.DataFrame(columns=['table_name', 'last_update', 'record_count', 'status'])

                df = pd.read_sql_query("""
                    SELECT table_name, last_update, record_count, status
                    FROM cache_metadata
                    ORDER BY 
                        CASE 
                            WHEN table_name LIKE 'dim_%' THEN 1 
                            WHEN table_name LIKE 'fato_%' THEN 2 
                            ELSE 3 
                        END,
                        table_name
                """, conn)

            return df

        except Exception as e:
            logger.error(f"Erro ao obter status do cache: {e}")
            # Retornar DataFrame vazio em caso de erro
            return pd.DataFrame(columns=['table_name', 'last_update', 'record_count', 'status'])


# Inicializar o gerenciador de dados
@st.cache_resource
def get_data_manager():
    with st.spinner('Conectando ao banco de dados...'):
        try:
            dm = DataManager()
            # GARANTIR que o cache seja inicializado
            dm.init_cache_db()
            return dm
        except Exception as e:
            st.error(f"Erro ao conectar: {e}")
            st.stop()


# Função para limpar cache se necessário
def reset_data_manager():
    """Força recriação do DataManager"""
    if 'data_manager' in st.session_state:
        del st.session_state['data_manager']
    get_data_manager.clear()
    dm = DataManager()
    dm.init_cache_db()
    return dm


# Interface principal
def main():
    if 'cache_updating' not in st.session_state:
        st.session_state.cache_updating = False
    # Verificar se DataManager tem os atributos necessários
    try:
        data_manager = get_data_manager()
        # Teste se tem o atributo equipes_excluidas
        _ = data_manager.equipes_excluidas
    except AttributeError:
        # Se não tem o atributo, recriar o DataManager
        st.warning("🔄 Atualizando configuração do sistema...")
        data_manager = reset_data_manager()
        st.success("✅ Sistema atualizado!")
        st.rerun()

    # Header
    st.markdown("""
    <div class="custom-header">
        <div class="logo-container">
            <img src="data:image/png;base64,{}" alt="Logo Rezende Energia" style="height: 80px; width: auto; filter: brightness(0) invert(1);">
        </div>
        <div class="text-container">
            <h1>⚡ BI Rezende Energia</h1>
            <p>Dashboard Executivo</p>
        </div>
    </div>
    """.format(get_base64_image(
        r"C:\Users\Pedro Curry\OneDrive\Área de Trabalho\Rezende\MARKETING\__sitelogo__Logo Rezende.png")),
        unsafe_allow_html=True)

    # Status do cache no sidebar
    st.sidebar.markdown("### 🗄️ Status do Cache Dimensional")

    # Mostrar equipes excluídas
    st.sidebar.markdown("#### ⚠️ Equipes Excluídas do Faturamento")
    equipes_excluidas_text = "<br>".join([f"• {eq}" for eq in data_manager.equipes_excluidas])
    st.sidebar.markdown(f"""
    <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 0.5rem; font-size: 0.8rem; color: #856404;">
        {equipes_excluidas_text}
    </div>
    """, unsafe_allow_html=True)

    cache_status = data_manager.get_cache_status()
    if not cache_status.empty:
        for _, row in cache_status.iterrows():
            last_update = datetime.fromisoformat(row['last_update']) if row['last_update'] else datetime.min
            time_ago = datetime.now() - last_update

            # Definir tempo de expiração baseado no tipo de tabela
            if row['table_name'].startswith('dim_'):
                expiry_hours = 168  # 7 dias para dimensões
            else:
                expiry_hours = data_manager.cache_duration_hours

            if time_ago > timedelta(hours=expiry_hours):
                status_class = "error"
                status_icon = "⚠️"
            else:
                status_class = ""
                status_icon = "✅"

            # Ícone baseado no tipo de tabela
            if row['table_name'].startswith('dim_'):
                table_icon = "📅" if "calendario" in row['table_name'] else "👥"
            else:
                table_icon = "📊"

            table_display = row['table_name'].replace('_', ' ').title()
            status_text = f"{status_icon} {table_icon} {table_display}: {row['record_count']:,} registros"

            st.sidebar.markdown(f"""
            <div class="cache-status {status_class}">
                {status_text}<br>
                <small>Atualizado: {last_update.strftime('%d/%m %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("Cache não inicializado")

    # Controles de cache
    st.sidebar.markdown("### 🔄 Controles de Cache")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Atualizar Cache", type="primary"):
            with st.spinner("Atualizando cache dimensional..."):
                results = data_manager.refresh_all_caches(force=True)

                for table, result in results.items():
                    if result['success']:
                        st.success(f"✅ {table.replace('_', ' ').title()}: {result['message']}")
                    else:
                        st.error(f"❌ {table.replace('_', ' ').title()}: {result['message']}")
                st.rerun()

    with col2:
        if st.button("🔧 Reset Sistema", help="Recarrega configurações"):
            data_manager = reset_data_manager()
            st.success("🔄 Sistema reiniciado!")
            st.rerun()

    auto_refresh = st.checkbox("Auto-refresh", value=True, help="Atualiza cache automaticamente quando necessário")

    # Inicializar no início do main()
    if 'cache_updating' not in st.session_state:
        st.session_state.cache_updating = False

    # Substituir o auto-refresh por:
    #if auto_refresh and not st.session_state.cache_updating:
        #needs_servicos, _ = data_manager.needs_refresh('fato_servicos')
        #if needs_servicos:
            #st.session_state.cache_updating = True
            #with st.spinner("Atualizando cache automaticamente..."):
               # data_manager.refresh_all_caches()
            #st.session_state.cache_updating = False
            #st.rerun()

    # Filtros baseados nas DIMENSÕES
    st.sidebar.markdown("### 🎛️ Filtros Dimensionais")

    # Carregar opções de filtro das DIMENSÕES
    filter_options = data_manager.get_dimension_filters()

    # Filtros de data
    st.sidebar.markdown("#### 📅 Período")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        data_inicio = st.date_input(
            "De:",
            value=datetime.now() - timedelta(days=30),
            key="data_inicio"
        )
    with col2:
        data_fim = st.date_input(
            "Até:",
            value=datetime.now(),
            key="data_fim"
        )

    # Filtros de equipe (da dimensão)
    st.sidebar.markdown("#### 👥 Equipes e Operação")
    encarregado = st.sidebar.multiselect(
        "Equipe:",
        options=filter_options.get('equipes', []),
        default=[]
    )

    tipo_servico = st.sidebar.multiselect(
        "Tipo de Serviço:",
        options=filter_options.get('tipos_servico', []),
        default=[]
    )

    placa = st.sidebar.multiselect(
        "Placa da Equipe:",
        options=filter_options.get('placas', []),
        default=[]
    )

    # Filtros de localização (da dimensão)
    st.sidebar.markdown("#### 🏢 Localização e Contrato")
    regional = st.sidebar.multiselect(
        "Regional:",
        options=filter_options.get('regionais', []),
        default=[]
    )

    nota = st.sidebar.multiselect(
        "Nota/PEP:",
        options=filter_options.get('notas', []),
        default=[]
    )

    # Construir filtros
    filters = {
        'data_inicio': data_inicio.isoformat(),
        'data_fim': data_fim.isoformat(),
        'equipes': encarregado,
        'tipos_servico': tipo_servico,
        'placas': placa,
        'regionais': regional,
        'notas': nota
    }

    # Carregar dados do cache dimensional
    with st.spinner("Carregando dados do modelo dimensional..."):
        df_main = data_manager.get_servicos_data(filters)
        df_meta = data_manager.get_producao_vs_meta_data(filters)
        df_faturamento = data_manager.get_faturamento_data(filters)

    if df_main.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        st.info("💡 Verifique se o cache foi inicializado e tente atualizar os dados.")
        return

    # Cards de métricas principais
    st.markdown("### 📊 Indicadores Principais")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        qtd_equipes = df_main['cod_equipe'].nunique()
        delta_equipes = f"+{int(qtd_equipes * 0.05)}" if qtd_equipes > 0 else "0"
        st.metric(
            label="👥 Equipes Ativas",
            value=f"{qtd_equipes:,}",
            delta=delta_equipes
        )

    with col2:
        valor_total = df_main['valor_total'].sum()
        delta_valor = f"+R$ {valor_total * 0.08:.0f}" if valor_total > 0 else "R$ 0"
        st.metric(
            label="💰 Faturamento Total",
            value=f"R$ {valor_total:,.2f}",
            delta=delta_valor
        )

    with col3:
        total_servicos = df_main['quantidade'].sum()
        delta_servicos = f"+{int(total_servicos * 0.12)}" if total_servicos > 0 else "0"
        st.metric(
            label="🔧 Total de Serviços",
            value=f"{total_servicos:,}",
            delta=delta_servicos
        )

    with col4:
        if total_servicos > 0:
            ticket_medio = valor_total / total_servicos
            delta_ticket = f"R$ {ticket_medio * 0.03:.2f}" if ticket_medio > 0 else "R$ 0"
            st.metric(
                label="📈 Ticket Médio",
                value=f"R$ {ticket_medio:.2f}",
                delta=delta_ticket
            )
        else:
            st.metric(label="📈 Ticket Médio", value="R$ 0,00", delta="R$ 0")

    # Gráficos
    st.markdown("### 📈 Análises Visuais")

    # Gráfico Produção vs Meta (SEM subplot - apenas faturamento vs meta)
    if not df_meta.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Título com indicação de filtro e totais
        periodo_texto = f"{data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}"
        total_equipes = len(df_meta)
        total_faturamento = df_meta['producao_real'].sum()
        total_meta = df_meta['meta_equipe'].sum()
        percentual_geral = (total_faturamento / total_meta * 100) if total_meta > 0 else 0

        st.markdown(f'''
        <div class="chart-title">
            🎯 Faturamento vs Meta por Equipe<br>
            <small style="font-size: 0.8em; opacity: 0.7;">
                Período: {periodo_texto} | {total_equipes} equipes<br>
                Faturamento: R$ {total_faturamento:,.2f} | Meta: R$ {total_meta:,.2f} | Atingimento: {percentual_geral:.1f}%
            </small>
        </div>
        ''', unsafe_allow_html=True)

        # Criar gráfico simples de barras agrupadas
        fig_meta = go.Figure()

        # Barras de faturamento
        fig_meta.add_trace(go.Bar(
            name='Faturamento',
            x=df_meta['equipe'],
            y=df_meta['producao_real'],
            marker_color='#F7931E',
            opacity=0.8,
            text=[f'R$ {x:,.0f}' for x in df_meta['producao_real']],
            textposition='outside',
            textfont=dict(size=11, color='#000000', family='Inter'),
        ))

        # Barras de meta se houver dados válidos
        if df_meta['meta_equipe'].sum() > 0:
            fig_meta.add_trace(go.Bar(
                name='Meta',
                x=df_meta['equipe'],
                y=df_meta['meta_equipe'],
                marker_color='#000000',
                opacity=0.6,
                text=[f'R$ {x:,.0f}' if x > 0 else 'Sem Meta' for x in df_meta['meta_equipe']],
                textposition='outside',
                textfont=dict(size=11, color='#000000', family='Inter'),
            ))

        fig_meta.update_layout(
            height=450,
            xaxis_title="Equipes",
            yaxis_title="Faturamento (R$)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.98,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            margin=dict(l=70, r=30, t=80, b=100),
            barmode='group',
            xaxis=dict(tickangle=45, showgrid=False),
            yaxis=dict(
                tickformat=",.0f",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.15)',
                rangemode='tozero'
            )
        )

        # Ajustar range para evitar corte de texto
        if not df_meta.empty:
            max_faturamento = df_meta['producao_real'].max()
            max_meta = df_meta['meta_equipe'].max() if df_meta['meta_equipe'].sum() > 0 else 0
            max_valor = max(max_faturamento, max_meta)
            fig_meta.update_yaxes(range=[0, max_valor * 1.20])

        st.plotly_chart(fig_meta, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # SEÇÃO SEPARADA: Gráfico de Percentual de Atingimento
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📊 Percentual de Atingimento da Meta por Equipe</div>',
                    unsafe_allow_html=True)

        fig_percentual = go.Figure()

        # Cores baseadas na performance
        colors = ['#4CAF50' if x >= 100 else '#FF9800' if x >= 80 else '#F44336' for x in df_meta['percentual_meta']]

        fig_percentual.add_trace(go.Bar(
            x=df_meta['equipe'],
            y=df_meta['percentual_meta'],
            marker_color=colors,
            text=[f'{x:.1f}%' for x in df_meta['percentual_meta']],
            textposition='outside',
            textfont=dict(size=11, color='#000000', family='Inter'),
            showlegend=False
        ))

        # Linha de referência 100%
        fig_percentual.add_hline(
            y=100,
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text="Meta 100%",
            annotation_position="top right"
        )

        fig_percentual.update_layout(
            height=350,
            xaxis_title="Equipes",
            yaxis_title="% de Atingimento da Meta",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            margin=dict(l=70, r=30, t=60, b=100),
            xaxis=dict(tickangle=45, showgrid=False),
            yaxis=dict(
                tickformat=".0f",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.15)',
                rangemode='tozero'
            )
        )

        # Ajustar range do percentual
        if not df_meta.empty:
            max_percentual = df_meta['percentual_meta'].max()
            fig_percentual.update_yaxes(range=[0, max(max_percentual * 1.15, 120)])

        st.plotly_chart(fig_percentual, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfico de Faturamento Diário
    if not df_faturamento.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        periodo_texto = f"{data_inicio.strftime('%d/%m/%Y')} a {data_fim.strftime('%d/%m/%Y')}"
        st.markdown(
            f'<div class="chart-title">💰 Faturamento Diário<br><small style="font-size: 0.8em; opacity: 0.7;">Período: {periodo_texto}</small></div>',
            unsafe_allow_html=True)

        fig_fat = px.bar(
            df_faturamento,
            x='data',
            y='faturamento_diario',
            title="",
            color='faturamento_diario',
            color_continuous_scale=['#000000', '#F7931E'],
            text='faturamento_diario'
        )

        fig_fat.update_traces(
            texttemplate='R$ %{text:,.0f}',
            textposition='outside',
            textfont=dict(size=12, color='#000000', family='Inter', weight='bold')
        )

        fig_fat.update_layout(
            height=400,
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            showlegend=False,
            xaxis=dict(tickangle=45),
            yaxis=dict(
                tickformat=",.0f",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.15)',
                rangemode='tozero'
            ),
            margin=dict(l=80, r=20, t=60, b=120)
        )

        if not df_faturamento.empty:
            max_value = df_faturamento['faturamento_diario'].max()
            fig_fat.update_yaxes(range=[0, max_value * 1.15])

        st.plotly_chart(fig_fat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Gráfico Top Equipes
    if not df_main.empty:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🏆 Top 10 Equipes por Faturamento</div>', unsafe_allow_html=True)

        equipes_top = df_main.groupby('nome_equipe').agg({
            'valor_total': 'sum',
            'quantidade': 'sum'
        }).reset_index().sort_values('valor_total', ascending=False).head(10)

        fig_top = px.bar(
            equipes_top,
            x='nome_equipe',
            y='valor_total',
            title="",
            color='valor_total',
            color_continuous_scale=['#000000', '#F7931E'],
            text='valor_total'
        )

        fig_top.update_traces(
            texttemplate='R$ %{text:,.0f}',
            textposition='outside',
            textfont=dict(size=12, color='#000000', family='Inter', weight='bold')
        )

        fig_top.update_layout(
            height=400,
            xaxis_title="Equipes",
            yaxis_title="Faturamento Total (R$)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            showlegend=False,
            xaxis=dict(tickangle=45),
            yaxis=dict(
                tickformat=",.0f",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.15)',
                rangemode='tozero'
            ),
            margin=dict(l=80, r=20, t=60, b=120)
        )

        if not equipes_top.empty:
            max_value = equipes_top['valor_total'].max()
            fig_top.update_yaxes(range=[0, max_value * 1.15])

        st.plotly_chart(fig_top, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabela detalhada
    st.markdown("### 📋 Tabela Detalhada de Operações")

    # Preparar dados da tabela
    df_table = df_main.copy()
    df_table['valor_unitario'] = df_table['valor_unitario'].round(2)
    df_table['valor_total'] = df_table['valor_total'].round(2)

    # Renomear colunas para exibição
    df_table_display = df_table[[
        'nome_equipe', 'descricao_servico', 'quantidade', 'valor_unitario', 'valor_total',
        'regional', 'nota', 'placa', 'cidade', 'data_execucao'
    ]].copy()

    df_table_display.columns = [
        'Equipe', 'Descrição do Serviço', 'Quantidade', 'Valor Unitário (R$)', 'Valor Total (R$)',
        'Regional', 'Nota', 'Placa', 'Cidade', 'Data'
    ]

    # Formatação da tabela
    def format_currency(val):
        return f"R$ {val:,.2f}"

    def format_date(val):
        if pd.isna(val):
            return "N/A"
        try:
            return pd.to_datetime(val).strftime("%d/%m/%Y")
        except:
            return str(val)

    # Aplicar formatações
    df_table_display['Valor Unitário (R$)'] = df_table_display['Valor Unitário (R$)'].apply(format_currency)
    df_table_display['Valor Total (R$)'] = df_table_display['Valor Total (R$)'].apply(format_currency)
    df_table_display['Data'] = df_table_display['Data'].apply(format_date)

    # Controles da tabela
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_term = st.text_input("🔍 Buscar na tabela:", placeholder="Digite para filtrar...")

    with col2:
        page_size = st.selectbox("Registros por página:", [10, 25, 50, 100], index=1)

    with col3:
        sort_column = st.selectbox("Ordenar por:", df_table_display.columns.tolist())

    # Filtrar tabela com busca
    if search_term:
        mask = df_table_display.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(
            axis=1)
        df_filtered = df_table_display[mask]
    else:
        df_filtered = df_table_display.copy()

    # Ordenar
    try:
        if sort_column in ['Valor Unitário (R$)', 'Valor Total (R$)']:
            # Para colunas monetárias, ordenar pelo valor original
            orig_col = 'valor_unitario' if 'Unitário' in sort_column else 'valor_total'
            sort_values = df_table[orig_col].loc[df_filtered.index]
            df_filtered = df_filtered.iloc[sort_values.argsort()[::-1]]
        else:
            df_filtered = df_filtered.sort_values(sort_column)
    except:
        pass  # Se der erro na ordenação, manter ordem original

    # Paginação
    total_pages = len(df_filtered) // page_size + (1 if len(df_filtered) % page_size > 0 else 0)
    if total_pages > 0:
        page = st.selectbox(f"Página (Total: {total_pages}):", range(1, total_pages + 1))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        df_page = df_filtered.iloc[start_idx:end_idx]
    else:
        df_page = df_filtered
        page = 1

    # Configurar colunas da tabela
    column_config = {
        "Equipe": st.column_config.TextColumn("Equipe", width="medium"),
        "Descrição do Serviço": st.column_config.TextColumn("Descrição do Serviço", width="large"),
        "Quantidade": st.column_config.NumberColumn("Quantidade", format="%d", width="small"),
        "Valor Unitário (R$)": st.column_config.TextColumn("Valor Unitário (R$)", width="medium"),
        "Valor Total (R$)": st.column_config.TextColumn("Valor Total (R$)", width="medium"),
        "Regional": st.column_config.TextColumn("Regional", width="medium"),
        "Nota": st.column_config.TextColumn("Nota", width="small"),
        "Placa": st.column_config.TextColumn("Placa", width="small"),
        "Cidade": st.column_config.TextColumn("Cidade", width="medium"),
        "Data": st.column_config.TextColumn("Data", width="small")
    }

    # Exibir tabela
    st.dataframe(
        df_page,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Informações da tabela e download
    if not df_filtered.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 **Total de registros:** {len(df_filtered):,}")
        with col2:
            st.info(f"📄 **Página atual:** {page} de {total_pages if total_pages > 0 else 1}")
        with col3:
            st.info(f"💾 **Registros na página:** {len(df_page):,}")

        # Estatísticas adicionais baseadas nos filtros aplicados
        if len(df_filtered) > 0:
            st.markdown("#### 📈 Estatísticas dos Dados Filtrados")

            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            with stats_col1:
                equipes_filtradas = df_filtered['Equipe'].nunique()
                st.metric("Equipes Únicas", equipes_filtradas)

            with stats_col2:
                valor_medio_filtrado = df_main.loc[
                    df_filtered.index if len(df_filtered) <= len(df_main) else df_main.index, 'valor_total'].mean()
                st.metric("Valor Médio", f"R$ {valor_medio_filtrado:.2f}")

            with stats_col3:
                servicos_filtrados = df_main.loc[
                    df_filtered.index if len(df_filtered) <= len(df_main) else df_main.index, 'quantidade'].sum()
                st.metric("Serviços Filtrados", f"{servicos_filtrados:,}")

            with stats_col4:
                regionais_filtradas = df_filtered['Regional'].nunique()
                st.metric("Regionais Ativas", regionais_filtradas)

        # Download da tabela
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Baixar Tabela Completa (CSV)",
            data=csv,
            file_name=f"rezende_energia_operacoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados.")

        # Sugestões quando não há dados
        st.markdown("""
        ### 💡 Sugestões:
        - Verifique se as datas estão corretas
        - Remova alguns filtros para ampliar a busca
        - Certifique-se de que existem dados no período selecionado
        - Clique em "Atualizar Cache" para buscar dados mais recentes
        """)

    # Debug/Status avançado
    with st.expander("🔍 Informações Técnicas do Sistema"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Dados Carregados:**")
            if not df_main.empty:
                data_min = df_main['data_execucao'].min()
                data_max = df_main['data_execucao'].max()
                st.write(f"• Registros: {len(df_main):,}")
                st.write(f"• Período: {data_min} a {data_max}")
                st.write(f"• Equipes: {df_main['cod_equipe'].nunique()}")

                # Verificar totais
                total_kpi = df_main['valor_total'].sum()
                total_grafico = df_meta['producao_real'].sum() if not df_meta.empty else 0
                diferenca = abs(total_kpi - total_grafico)
                consistencia = "✅ Consistente" if diferenca < 1 else "⚠️ Diferença detectada"
                st.write(f"• Consistência: {consistencia}")

                if not df_meta.empty:
                    equipes_com_meta = df_meta[df_meta['meta_equipe'] > 0][
                        'equipe'].nunique() if 'meta_equipe' in df_meta.columns else 0
                    st.write(f"• Equipes c/ Meta: {equipes_com_meta}")
            else:
                st.write("• Nenhum dado carregado")

        with col2:
            st.markdown("**🎛️ Filtros Aplicados:**")
            filtros_ativos = []
            if filters.get('data_inicio') or filters.get('data_fim'):
                filtros_ativos.append(f"📅 Data: {data_inicio.strftime('%d/%m')} - {data_fim.strftime('%d/%m')}")
            if filters.get('equipes'):
                filtros_ativos.append(f"👥 Equipes: {len(filters['equipes'])}")
            if filters.get('tipos_servico'):
                filtros_ativos.append(f"🔧 Serviços: {len(filters['tipos_servico'])}")
            if filters.get('placas'):
                filtros_ativos.append(f"🚗 Placas: {len(filters['placas'])}")
            if filters.get('regionais'):
                filtros_ativos.append(f"🏢 Regionais: {len(filters['regionais'])}")
            if filters.get('notas'):
                filtros_ativos.append(f"📋 Notas: {len(filters['notas'])}")

            if filtros_ativos:
                for filtro in filtros_ativos:
                    st.write(f"• {filtro}")
            else:
                st.write("• Nenhum filtro ativo")

    # Informações do cache no rodapé
    st.markdown("---")

    # Status detalhado do cache dimensional
    with st.expander("📊 Status Detalhado do Cache Dimensional"):
        cache_status = data_manager.get_cache_status()
        if not cache_status.empty:
            # Adicionar ícones e formatação
            cache_display = cache_status.copy()
            cache_display['Tipo'] = cache_display['table_name'].apply(lambda x:
                                                                      "📅 Dimensão" if x.startswith(
                                                                          'dim_') else "📊 Fato")
            cache_display['Tabela'] = cache_display['table_name'].apply(lambda x:
                                                                        x.replace('_', ' ').title())
            cache_display['Registros'] = cache_display['record_count'].apply(lambda x: f"{x:,}")
            cache_display['Última Atualização'] = cache_display['last_update'].apply(lambda x:
                                                                                     datetime.fromisoformat(x).strftime(
                                                                                         '%d/%m/%Y %H:%M') if x else 'Nunca')

            st.dataframe(
                cache_display[['Tipo', 'Tabela', 'Registros', 'Última Atualização', 'status']],
                column_config={
                    "Tipo": "Tipo",
                    "Tabela": "Tabela",
                    "Registros": "Registros",
                    "Última Atualização": "Última Atualização",
                    "status": "Status"
                },
                hide_index=True,
                use_container_width=True
            )

        # Informações do banco SQLite
        try:
            db_size = os.path.getsize(data_manager.db_path) / (1024 * 1024)  # MB
        except:
            db_size = 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tamanho do Cache", f"{db_size:.2f} MB")
        with col2:
            total_tables = len(cache_status) if not cache_status.empty else 0
            st.metric("Tabelas no Cache", total_tables)
        with col3:
            total_records = cache_status['record_count'].sum() if not cache_status.empty else 0
            st.metric("Total de Registros", f"{total_records:,}")

    # Footer
    st.markdown(
        "<div style='text-align: center; color: #6c757d; font-size: 0.9rem; margin-top: 2rem;'>"
        "⚡ <b>BI - Rezende Energia</b><br>"
        f"🕒 Última atualização: {datetime.now().strftime('%d/%m/%Y às %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
