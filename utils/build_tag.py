class Tag(object):
    def __init__(self):
        self.static_tags = self.__load_static_tags()
        self.id2tags = self.__load_id2tags()
        self.tags2id = self.__load_tags2id()

    def array2tags(self, array):
        tags = []
        for id in array:
            tags.append(self.id2tags[id])
        return tags

    def tags2array(self, tags):
        array = []
        for tag in self.static_tags:
            if tag in tags:
                array.append(1)
            else:
                array.append(0)
        return array

    def inv_tags2array(self, array):
        tags = []
        for i, value in enumerate(array):
            if value != 0:
                tags.append(self.id2tags[i])
        return tags

    def __load_id2tags(self):
        id2tags = {}
        for i, tag in enumerate(self.static_tags):
            id2tags[i] = tag
        return id2tags

    def __load_tags2id(self):
        tags2id = {}
        for i, tag in enumerate(self.static_tags):
            tags2id[tag] = i
        return tags2id

    def __load_static_tags(self):
        static_tags_name = ['normal','cervical vertebrae','atelectases','atelectasis',
                            'plate-like atelectasis','pulmonary atelectasis','degenerative change','aorta, thoracic',
                            'focal atelectasis','scarring','pericardial effusion','pleural effusions','pleural effusion',
                            'osteophytes','thoracic vertebrae','malignancy','pneumonia','bronchitis','calcified granuloma',
                            'nodule','granuloma','atheroscleroses','emphysemas','aorta','atherosclerosis','emphysema',
                            'pulmonary emphysema','prostheses','stents','thoracic aorta','opacity','chronic interstitial lung disease',
                            'scar','lung diseases, interstitial','scars','granulomas','edemas','cardiomegaly','edema',
                            'granulomatous disease','pneumoperitoneum','abdominal surgery','pleural thickening','pulmonary disease','copd',
                            'lung diseases, obstructive','pulmonary disease, chronic obstructive','infiltrates','effusion',
                            'catheters','bulla','lung neoplasms','congestion','pulmonary edema','cholecystectomies','cholecystectomy',
                            'clip','metastatic disease','sclerotic','artifacts','lung granuloma','infection','thoracic spondylosis',
                            'subcutaneous emphysema','scolioses','scoliosis','stent','deformity','lumbar vertebrae','closure','dish',
                            'right upper lobe pneumonia','interstitial lung disease','lung surgeries','sutures','calcinosis','scleroses',
                            'sternotomy','humerus','fracture','hyperinflation lungs','humeral fractures','pathologic fractures','catheterization',
                            'catheterization, central venous','fractures, pathologic','adenopathy','lymph nodes','osteopenia','rib fracture',
                            'fractures, bone','rib fractures','pacemaker, artificial','collapse','pleural diseases','cervical spine fusion','sarcoidoses',
                            'tortuous aorta','aortic diseases','sarcoidosis','diaphragm','cavitation','tuberculoses','hydropneumothorax',
                            'tuberculosis','central venous catheters','central venous catheter','right atrium','large hiatal hernia','spine',
                            'bilateral pleural effusion','spinal osteophytosis','shoulder','surgical clip','chronic granulomatous disease','ribs',
                            'hyperexpansion','hernia, hiatal','aortic calcifications','ectasia','pulmonary hypertension','pulmonary artery',
                            'intubation, gastrointestinal','rib','callus','lymphadenopathy','patchy atelectasis','air trapping','pleural fluid','heart atria',
                            'diaphragms','aortic ectasia','cicatrix','surgery','hiatal hernia','arthritic changes','fibroses','fibrosis',
                            'calcifications of the aorta','hyperinflation','obstructive pulmonary diseases','dilatation, pathologic','spondylosis',
                            'eventration','renal osteodystrophies','lymph','bilateral hilar adenopathy','bypass grafts','thoracotomies','coronary artery bypass',
                            'thoracotomy','bullae','right-sided pleural effusion','fusion','old injury','spinal fractures','foreign body','pneumothorax','degenerative disc diseases','intervertebral disc degeneration','chronic lung disease',
                            'chronic obstructive pulmonary disease','hilar calcification','cabg','air','gases','ascending aorta','lymphatic diseases',
                            'mediastinal diseases','atherosclerotic vascular disease','multiple pulmonary nodules','tracheostomies','tracheostomy',
                            'displacement','picc','pleura','goiter','hyperlucent lung','dialysis','renal dialysis','postoperative period',
                            'aspiration','lung disease','lung diseases','hilar adenopathy','calcified lymph nodes','vascular calcification','abdomen',
                            'pulmonary arterial hypertension','hypertension, pulmonary','middle lobe syndrome','catheter','jugular veins','bronchiectases','bronchiectasis','left upper lobe pneumonia','pulmonary arteries','cervical fusion',
                            'spinal fusion','aortic valve replacement','aortic valve','heart valve prosthesis','heart valve prosthesis implantation',
                            'kyphosis','volume overload','degenerative joint disease','osteoarthritis','venous hypertension','hypertension','left atrial enlargement',
                            'right atrial enlargement','anchors','bronchopleural fistula','empyema','bronchial fistula','lymph node','clavicle','centrilobular emphysema','mitral annular calcification','histoplasmoses','histoplasmosis',
                            'bronchiolitides','bronchiolitis','granulomatous infection','degenerative disease','pneumonitis','neoplasm','nipple',
                            'chronic disease','aortic atherosclerosis','lobectomy','bone demineralization','hypoventilation','inflammation','pleural plaque',
                            'discoid atelectasis','exostoses','hiatus hernia','hernia, hiatus','intubation, intratracheal','interstitial disease','trachea',
                            'pulmonary fibroses','pulmonary fibrosis','descending aorta','congestive heart failure','pectus carinatum','funnel chest','kyphoses',
                            'pulmonary artery hypertension','cardiac monitor','azygos lobe','interstitial pulmonary edema','diffuse idiopathic skeletal hyperostosis','hyperostosis, diffuse idiopathic skeletal','hypertrophy','cysts',
                            'ascending aortic aneurysm','aortic aneurysm','aortic aneurysm, thoracic','enlarged heart','vertebroplasty','non-displaced fracture',
                            'histoplasmoma','breast carcinoma','bullous emphysema','obstructive lung disease','heart failure','aneurysm','lumpectomy','thoracolumbar scoliosis',
                            'clavicular fracture','right ventricle','pneumonectomy','hernias','stomach','diaphragm eventration','aorta tortuous','pectus excavatum','mastectomies',
                            'respiratory tract diseases','pulmonary valve','apical granuloma','humeral head','demineralization','picc line','right lower lobe pneumonia',
                            'spine curvature','multiple myeloma','left lower lobe pneumonia','chest tubes','mass lesion','piercing','humeral fracture','osteophyte','sternum',
                            'displaced fractures','vena cava, superior','bone density','thoracic wall','left-sided pleural effusion','lung hyperinflation','cystic fibrosis',
                            'aneurysms','pulmonary granuloma',
                            'gunshot wounds','foreign bodies','adipose tissue','mid sternotomy','catheterization, peripheral','aortic dissection','aneurysm, dissecting',
                            'bone diseases, metabolic','vascular calcifications','thorax','lung cancer','spondylarthritis','tips','atypical pneumonias','breast implants',
                            'cervical arthritis','lymphomas','lymphoma','coronary vessels','unknown']


        return static_tags_name
