# Previous requirements
- Setup a google service account with enough permissions to use buckets.
- Download the google_service_account.json file, rename it and put on the root directory.
- Create a bucket with the next folder structure:
    - bucket/
        - input/
        - output/
        - template_file
- The diploma template need to aruco markers, you can find a template in assets direcotory. 
# Some useful commands
### Build docker image
```bash
 docker build -t diploma-maker .  
```

### Run container
```bash
docker run -d --name diploma-maker -p 80:80 diploma-maker 
```

### Test project
In your browser go to http://localhost/docs#


# Deploy on Google Cloud Run
```bash 
gcloud run deploy --source .
```

## Result 
You can see an example in assets directory