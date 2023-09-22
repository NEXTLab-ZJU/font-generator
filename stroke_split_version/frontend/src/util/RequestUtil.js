const axios = require('axios');
const config = require('../config');
class RequestUtil {
    static async getCurrentProgress() {
        return axios({
            method: 'get',
            url: `${config.request_url}/progress`,
            data: {

            }
        });
    }

    static async getCurrentVectorProgress() {
        return axios({
            method: 'get',
            url: `${config.request_url}/vector_progress`,
            data: {

            }
        });
    }

    static async start(path, json_path) {
        return axios({
            method: 'post',
            url: `${config.request_url}/start`,
            data: {
                path,
                json_path
            }
        });
    }

    static async getResultDir() {
        return axios({
            method: 'get',
            url: `${config.request_url}/result_dir`,
            data: {

            }
        });
    }
}


module.exports = RequestUtil