import sqlalchemy as SA
import psycopg2
import pandas as pd
import config as config


def get_redshift_dataframe():
    host = config.db_host
    username = config.db_username
    password = config.db_password
    port = config.db_port
    db = config.db_database
    url = "{d}+{driver}://{u}:{p}@{h}:{port}/{db}".format(d="redshift",
                                                          driver='psycopg2',
                                                          u=username,
                                                          p=password,
                                                          h=host,
                                                          port=port,
                                                          db=db)
    engine = SA.create_engine(url)
    cnn = engine.connect()

    strSQL = "select * from ds.mambu_loanaccount limit 1000"
    try:
        rows = cnn.execute(strSQL)
        list_of_dicts = [dict((key, value) for key, value in row.items()) for row in rows]
        df = pd.DataFrame(list_of_dicts)
        return df
    except:
        raise
