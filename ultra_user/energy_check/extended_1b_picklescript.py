import ultra_user.energy_check.event_dataset as ed
import ultra_user.pipeline.test_data as td
import ultra_user.energy_check.de_extended_calcs as de_calcs
import pickle

d_csv = ed.EventDataset()
l1a = td.de_dataset()

l1b=de_calcs.get_1bdict(l1a)
#pkldir = '/Users/demajr1/tmp/'
pkldir = 'data/pkls/'


pkfile = open(pkldir+'dataset_from_csv.pkl',"wb")
pickle.dump(d_csv,pkfile)
pkfile.close()

pkfile = open(pkldir+'dataset_l1a.pkl',"wb")
pickle.dump(l1a,pkfile)
pkfile.close()

pkfile = open(pkldir+'dataset_l1b.pkl',"wb")
pickle.dump(l1b,pkfile)
pkfile.close()

