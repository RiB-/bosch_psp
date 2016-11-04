"""
  **************************************
  Created by Romano Foti - rfoti
  On 09/15/2016
  *************************************
"""

#******************************************************************************
# Importing packages
#******************************************************************************
#-----------------------------
# Standard libraries
#-----------------------------
import requests
import base64
import zipfile

#******************************************************************************
# Defining functions
#******************************************************************************

#******************************************************************************

#******************************************************************************
# Defining classes
#******************************************************************************

class KaggleRequest():
    '''
    Connects to Kaggle and downloads datasets.
    '''

    def __init__(self, credentials_file=None):
        if not credentials_file:
            self.credentials_file = './kaggle_cred.cred'
        #end
    #end

    def decrypt(self, credentials_file):
        '''
        This function retrieves the encrypted credential file
        and returns a dictionary with username and password
        '''
        cred_file = open(credentials_file, 'r')
        cred_lines_encry_ls = cred_file.read().split(',')
        try:
            creds_dc = {'UserName': base64.b64decode(cred_lines_encry_ls[0]), 
                        'Password': base64.b64decode(cred_lines_encry_ls[1])}
        except:
            print 'Problem decrypting credentials. Request terminated.'
            return
        #end
        return creds_dc
    #end

    def unzip(self, filename):
        output_path = '/'.join([level for level in filename.split('/')[0:-1]]) + '/'
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(output_path)
        #end
        z.close()
        print 'File successfully unzipped!'
        return
    #end

    def retrieve_dataset(self, data_url, local_filename=None, chunksize=512, unzip=True):
        '''
        Connects to Kaggle website, downloads the dataset one chunk at a time
        and saves it locally.
        '''
        if not data_url:
            print 'agfag'
        if not local_filename:
            try:
                local_filename = './' + data_url.split('/')[-1]
                print 'Dataset name inferred from data_url. It is going to be saved in the default location.'
            except:
                print 'Could not infer data name, request terminated.'
                return
            #end
        #end
        kaggle_info = self.decrypt(self.credentials_file)
        chunks = chunksize * 1024
        req = requests.get(data_url) # attempts to download the CSV file and gets rejected because we are not logged in
        req = requests.post(req.url, data=kaggle_info, stream=True) # login to Kaggle and retrieve the data
        f = open(local_filename, 'w')
        for chunk in req.iter_content(chunk_size=chunks): # Reads 512KB at a time into memory
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
            #end
        #end
        f.close()
        print 'Data successfully downloaded!'
        if unzip:
            self.unzip(local_filename)
        #end
        return
    #end

#end