# Previous requirements
- Setup a google service account with enough permissions to use buckets
- Create a bucket with the next folder structure:
    - bucket/
        - input/
        - output/
        - template_file

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