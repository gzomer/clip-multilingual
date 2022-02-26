const fs = require('fs')
const axios = require('axios')
const crypto = require('crypto')
const randomUseragent = require('random-useragent');
const async = require('async')

// Initialize constants
const datasetFolder = './datasets/googlecc'
const TIMEOUT = 5000
const PARALLEL_DOWNLOAD = 50
const SAVE_EVERY = 10000
const LOG_EVERY = 1000
const processedFile = `${datasetFolder}/processed_map.json`
const imagesFolder = `${datasetFolder}/images/`

function loadCaptionsAndURLS(file) {
    let data = fs.readFileSync(file, 'utf8')
    let lines = data.split('\n')
    let headers = lines[0].split('\t')
    return lines.slice(1).map(line => {
        let row = {}
        let values = line.split('\t')
        for (let i = 0; i < headers.length; i++) {
            row[headers[i]] = values[i]
        }
        return row
    })
}

function sha1(text) {
    return crypto.createHash('sha1').update(text).digest('hex')
}

async function downloadImage(row) {
    counter += 1

    let url = row['url']
    if (url == '' || typeof url == 'undefined' || url == null) {
        return
    }
    if (counter % SAVE_EVERY == 0) {
        fs.writeFileSync(processedFile, JSON.stringify(processedMap, null, 2))
    }

     if (counter % LOG_EVERY == 0) {
        let now = new Date()
        let duration = now - start
        let averageDuration = duration / Object.keys(processedMap).length
        console.log(`Processed ${Object.keys(processedMap).length}, Success ${Object.keys(successMap).length}, Avg duration ${averageDuration}`)
    }
    let filename = sha1(url)+ '.jpg'
    if (typeof processedMap[url] !== 'undefined') {
        if (processedMap[url] == 'success') {
            successMap[url] = true
        } else {
            errorsMap[url] = true
        }
        return
    }
    if (fs.existsSync(imagesFolder + filename)) {
        processedMap[url] = 'success'
        successMap[url] = true
        return
    }
    try {
        let headers = {
            'User-Agent': randomUseragent.getRandom()
        }
        let r = await axios.get(url, {responseType: 'arraybuffer', headers: headers, timeout: TIMEOUT})
        if (r.headers['content-type'].indexOf('image') != -1) {
            if (r.status == 200) {
                fs.writeFileSync(imagesFolder + filename, r.data)
                processedMap[url] = 'success'
                successMap[url] = true
            }
        } else {
            processedMap[url] = 'not_image'
        }
    } catch (e) {
        processedMap[url] = 'bad_request'
    }
}

// Init variables
let successMap = {}
let errorsMap = {}
let processedMap = {}
let start = new Date()
let counter = 0

function downloadDataset(datasetFile) {
    let dataset = loadCaptionsAndURLS(datasetFile)

    if (fs.existsSync(processedFile)) {
        processedMap = JSON.parse(fs.readFileSync(processedFile, 'utf8'))
    }

    async.mapLimit(dataset, PARALLEL_DOWNLOAD, downloadImage, (err, results) => {
        if (err) {
            console.log(err)
        }
        fs.writeFileSync(processedFile, JSON.stringify(processedMap, null, 2))
    })
}

downloadDataset(`${datasetFolder}/googlecc.tsv`)