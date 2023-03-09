from flask import Flask
from flask_cors import CORS
# 初始化所有应用
def create_app():
    import models, routes, services, conf
    app = Flask(__name__)
    CORS(app, resources="/*") #开启跨域
    app.config.from_object(conf.config.app_config)
    #models.init_app(app)
    routes.init_app(app)
    #services.init_app(app)
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', debug=True,port=5000)
