import os
import re
import cv2
import pandas as pd
import numpy as np
import json
import csv
import re
import warnings

class Database:
    def __init__(self, videos_path = None, motion_path = None, eegs_path = None, naming_conventions = None):
        self.videos_path = videos_path
        if(type(videos_path) == type(None)):
            self.videos_path = "/data/p_01888/Databook_cleaning/Video/"

        self.motion_path = motion_path
        if(type(motion_path) == type(None)):
            self.motion_path = '/data/pt_01888/motionData/'

        self.eegs_path = motion_path
        if(type(motion_path) == type(None)):
            self.eegs_path = '/data/p_01888/Databook_cleaning/EEG/'

        self.json_filename = "database.json"
        self.csv_filename = "database.csv"
        self.naming_conventions = naming_conventions


    def init_via_videos(self):
        """ Search for videos and append them to a dictionary structure.
            Use this structure as basis for the database.
        """
        if(type(self.videos_path)==type(None)):
            print("Set videospath first")
        self.dictionary = Database.compute_dict(self.videos_path,
                                                regex_file = self.naming_conventions["regex_video_file"],
                                                regex_folder = self.naming_conventions["regex_video_folder"],
                                                subtree_key = "video")

    def save_as_json(self):
        """ Saves the database dictionary as a .json file """

        with open(self.json_filename, 'w') as outfile:
            json.dump(self.dictionary, outfile)

    def load_json(self, filepath = None):
        """ Loads the database from as .json file """
        if(type(filepath) == type(None)):
            filepath = self.json_filename
        with open(filepath, 'r') as file:
            self.dictionary = json.load(file)

    def save_as_csv(self):
        """ Saves as Comma seperated file """
        dataframe = self.get_dataframe()
        dataframe.to_csv(self.csv_filename, sep='\t', encoding='utf-8')

    @staticmethod
    def n_frames(path):
        """ Obtain number of frames from videos using CV2 """
        cap = cv2.VideoCapture(path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frames

    def get_dict(self):
        """ Getter function for the dictionary that represents the database"""
        return self.dictionary

    def get_keys_of_level(self, lvl = 0, dictionary = None):
        """ Returns all keys for a given level in the dictionary"""
        #We always call with subtrees thus we need subfunction rec_keys_of_level
        if(dictionary==None):
            return self.rec_keys_of_level(self.dictionary, lvl)
        else:
            return self.rec_keys_of_level(dictionary, lvl)

    def rec_keys_of_level(self, subtree,lvl):
        """ Recursive helper method of get_keys_of_level(...)"""
        #if level is 0 return list of get_keys
        if(lvl <= 0):
            ret_lst = []
            try:
                for key, value in subtree.items():
                    ret_lst.append(key)
            except:
                pass
            return ret_lst
        else:
            ret_lst = []
            try:
                for key, value in subtree.items():
                    if(isinstance(value, dict)):
                        ret_lst.extend(self.rec_keys_of_level(value, lvl-1))
            except:
                pass
            return ret_lst

        #otherwise call recursively and return result

    @staticmethod
    def compute_dict(path, regex_file = ".*", regex_folder = ".*", subtree_key = "datatype_x"):
        """ Computes the dictionary by checking for valid video files in path and subdirectories.
            Creates a nested dictionary for all dyads and strores filepathes within.
            Path specifies a folder that contains subfolders
            Only files are processed that match the syntax exemplified by coSMIC_all_P01.wmv
            The dict may be accessed e.g. by returned_dict[0]["video"][2]["path"]
        """
        for root, dirs, files in os.walk(path):#Go through all dirs and files in current directory
            #foldername = root.split(os.sep)[-1] #isolate last part of path
            directories = {}

            sorted_files = [f for f in files]#Sort files by name before beginning
            sorted_files.sort()

            for file in sorted_files:# Go through files and check if they are a valid video
                if(re.match(regex_file, file)):

                    try:
                        attributes = {}
                        attributes["path"] = root + os.sep +file # os.sep equals / or \ respectively on UNIX and Windows

                        camera = re.findall("[0-9]", file)[-1]#Last integer in filename is assumed to specify camera!!
                        directories[camera] = attributes
                    except:
                        warnings.warn("File " + str(file) +" in directory " + str(root+d) +  " didn't match convention" )
                        pass #Regex didn't match any files

            sorted_dirs = [d for d in dirs]# sort directories before beginning
            sorted_dirs.sort()

            for d in sorted_dirs:#Append an int for each dyad
                try:
                    contents = {}
                    if(re.match(regex_folder, d)):
                        try:
                            pair = int("".join(re.findall("[0-9]", d)))
                        except:
                            pass

                        directories[pair] = {}
                        directories[pair][subtree_key] = Database.compute_dict(root+d, regex_file, regex_folder)#Recursive call
                except:
                    warnings.warn("Folder " + str(d) +" in source directory " + str(root+d) +  " didn't match convention" )
                    pass
            return directories

    def integrate_framenumbers(self):
        for trial, v in self.dictionary.items():
            for datatype, v1 in self.dictionary[trial].items():
                if datatype == "video":
                    for n_vid, v2 in self.dictionary[trial]["video"].items():
                        try:
                            filepath = self.dictionary[trial]["video"][n_vid]["path"]
                            self.dictionary[trial]["video"][n_vid]["n_frames"] = self.n_frames(filepath)
                        except:
                            raise Exception("Couldn't retrieve framenumber for file ")


    def get_dataframe(self):
        """ Get a pandas dataframe representation of the database.
            Convert self.dictionary to a dataframe
        """
        d = self.dictionary
        pairs = [key for key in d.keys()]
        self.un_id = [] # Will store e.g. [['video', 1, 'path'], ['video', 1, 'n_frames']...]

        for value in d.values():#Toplevel will make the rows
            self.acc_rec(value, [])# Retrieve keys to final values in the nested structure e.g [['video', 1, 'path'],...]

        #Make em unique i.e. avoid having same list twice in outer list
        self.un_id = [list(x) for x in set(tuple(i) for i in self.un_id)]
        self.un_id.sort()


        headers = ['.'.join([str(c) for c in x]) for x in self.un_id]#Get a string representation of each sublist

        ndarray = np.ndarray((max(pairs),len(self.un_id)), dtype=object)#Rows x columns

        for y in range(ndarray.shape[0]):
            for x in range(ndarray.shape[1]):
                val = None# FOR DEEPER NESTINGS ADJUST HERE:
                try: # select line by y i.e. first level entry in dict. Within subtree: Get
                    attr = self.un_id[x]

                    if(len(attr)==5):#e.g. ['eeg', 'metainfo', 'type', 48]
                        val = d[y][attr[0]][attr[1]][attr[2]][attr[3]][attr[4]]
                    if(len(attr)==4):#e.g. ['eeg', 'metainfo', 'type', 48]
                        val = d[y][attr[0]][attr[1]][attr[2]][attr[3]]
                    if(len(attr)==3):#e.g.[motion,1,path]
                        val = d[y][attr[0]][attr[1]][attr[2]]
                    if(len(attr)==2):#e.g. ['eeg', 'path']
                        val = d[y][attr[0]][attr[1]]
                    if(len(attr)==1):
                        val = d[y][attr[0]]

                except:# Sometimes values are not present because d[y] is none i.e. pair data is missing
                    ndarray[y][x] = None
                ndarray[y][x] = val

        dataframe = pd.DataFrame(ndarray, columns=headers)

        return dataframe


    def acc_rec(self, node, prefix):
        """
            Appends lists of keys e.g. ['video', 1, 'path'] or ['eeg', 'metainfo', 'channel', 3]
                    to self.un_id recursively
        """
        for key, value in node.items():
            if(isinstance(value, dict)):
                new =[]
                new.extend(prefix)
                new.append(key)
                self.acc_rec(value, new)
            else:
                new = []
                new.extend(prefix)
                new.append(key)
                self.un_id.append(new)

    def integrate_motion(self):
        """
            Check .mot if file exists for given video and naming conventions.
            Add filepath of .mot data.
        """
        try:
            prefix = self.naming_conventions["motion"][0]
            infix = self.naming_conventions["motion"][1]
            postfix = self.naming_conventions["motion"][2]
        except:
            raise ValueError("No (valid) naming convention for 'motion' found. Expected a list of strings for prefix, infix and suffix")
            return

        d = self.dictionary
        for pair, v in list(d.items()):#Go down in dict tree (use list when modifying during iteration)
            for video, v1 in list(d[pair].items()):
                for n_video, v2 in list(d[pair][video].items()):

                    path = self.motion_path + prefix + str(pair) + infix + str(n_video)+ postfix
                    if(os.path.isfile(path)):
                        self.create_keys(self.dictionary,[pair,video,n_video,"motion","path"])
                        self.dictionary[pair][video][n_video]["motion"]["path"] = path


    def integrate_raw_eegs(self):
        """ Check if EEG files (.eeg and .vmrk) exist for given video and naming conventions.
            Load data if found and add filepath of eeg.
        """
        try:
            eeg_prefix = self.naming_conventions["eeg"][0]
            eeg_suffix = self.naming_conventions["eeg"][1]
            vmrk_prefix = self.naming_conventions["eeg_vmrk"][0]
            vmrk_suffix = self.naming_conventions["eeg_vmrk"][1]


        except:
            raise ValueError("No (valid) naming convention for 'eeg' found. Expected a list of strings for prefix and suffix")
            return

        d = self.dictionary
        for pair, v in list(d.items()):
            for isvideo, v1 in list(d[pair].items()):
                for n_video, v2 in list(d[pair][isvideo].items()):
                    path = self.eegs_path +eeg_prefix+str(pair)+eeg_suffix
                    path1 = self.eegs_path + vmrk_prefix + str(pair) + vmrk_suffix

                    if(os.path.isfile(path) and os.path.isfile(path1)):
                        self.dictionary[pair]["eeg"] = {}
                        self.dictionary[pair]["eeg"]["path"] = path
                        self.dictionary[pair]["eeg"]["metainfo"] = Database.parse_vmrk(path1)


    search_results = []
    @staticmethod
    def search_key(dictionary, search_key):
        Database.search_results = []
        Database.search_key_rec(dictionary, search_key, path = [])
        return Database.search_results

    @staticmethod
    def search_key_rec(root, search_key, path = []):
        if  isinstance(root, dict):
            for key, value in root.items():
                    current_path = path.copy()
                    current_path.append(key)

                    if key == search_key:
                        Database.search_results.append(current_path)
                    Database.search_key_rec(value, search_key, current_path)


    @staticmethod
    def parse_vmrk(path):
        """ Parses vmrk file and returns a dictionary containing the information.
            The keys denote the kind of data whereas the values are a dictionary
        """

        with open(path) as f:
            content = f.readlines()

        data = {'marker number':[], 'type':[], 'description':[], 'position':[], 'size':[], 'channel':[]}

        entry = 0
        for line in content:
            match = re.match("Mk", line)
            if(match != None):
                markers = re.search("[0-9][0-9]?", line)
                data["marker number"].append(int(markers.group(0)))
                line = line[markers.end():]#use rest of line only next

                markers = re.match("(.*?),",line)
                data["type"].append(markers.group(1)[1:])#Group 1 is exclusive , while group 0 is inclusive ,
                line = line[markers.end():]

                markers = re.search("(.*?),",line)
                data["description"].append(markers.group(1))
                line = line[markers.end():]

                markers = re.search("(.*?),",line)
                data["position"].append('0' + markers.group(1))# '0' + is necessary as some fields are empty
                line = line[markers.end():]

                markers = re.search("(.*?),",line)
                data["size"].append(int('0' + markers.group(1)))
                line = line[markers.end():]

                try:#In the first line there is an additional value we dont want to parse
                    data["channel"].append(int('0' + line))
                except:
                    data["channel"].append(0)
        return data

    def create_keys(self, dictionary, list_of_keys, pos = 0):#e.g. ['1']['motion']['in_roi']['1']
        """
            Creates keys if necessary (path in tree to leave node) such that one may easily add a values (leave)
        """
        if not pos == len(list_of_keys):
            try:#check if exists
                dictionary[list_of_keys[pos]]
            except:
                dictionary[list_of_keys[pos]] = {}

            self.create_keys(dictionary[list_of_keys[pos]], list_of_keys, pos+1)

    @staticmethod
    def max_key(keys):
        """
            Returns the key with the highest value assuming it is an int or string represnentation osf an int
        """
        roi_id = [int(x) for x in keys] #ids will be increasing integers
        roi_id.sort()
        if roi_id == []:#Obtain the highest integer
            current_id = '1'
        else:
            current_id = max(roi_id)
        return current_id

    def add_roi(self, dyad, video_number, start_frame, coordinates, mother = False, child = False, comment = ""):
        """
            Adds the information of a region of interest
        """

        if mother == False and child == False:
            raise ValueError("Either mother or child has to be True")
        if mother == True and child == True:
            raise ValueError("Either mother or child has to be False")

        dyad = str(dyad)
        video_number = str(video_number)

        self.create_keys(self.dictionary,[dyad,'video',video_number,'motion','in_roi'])
        c_id = self.max_key(self.get_dict()[dyad]['video'][video_number]['motion']['in_roi'].keys())

        self.dictionary[dyad]['video'][video_number]['motion']['in_roi'][c_id] = {"start_frame": start_frame,
                                                                        "coordinates" : coordinates,
                                                                        "mother": mother,
                                                                        "child": child,
                                                                        "comment": comment}
