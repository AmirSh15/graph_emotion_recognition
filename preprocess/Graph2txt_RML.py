import numpy as np

tr_feat = np.load('./train_graph_data_RML.npy')
tr_label = np.load('./train_graph_label_RML.npy')

tr_feat = np.reshape(tr_feat, newshape=(tr_feat.shape[0],
                                        tr_feat.shape[1], 68*2))


txt = []

txt.append(tr_feat.shape[0])
for i in range(tr_feat.shape[0]):
    for j in range(tr_feat.shape[1]):
        if(j==0):
            txt.append('%s %s'% (tr_feat.shape[1], int(tr_label[i,0])))
        if(j==tr_feat.shape[1]-1    ):
            a = [int(e) for e in tr_feat[i, j]]
            txt.append('%s 1 %s ' % (j, j -1) + ' '.join(str(e) + ' ' for e in a))
        else:
            a = [int(e) for e in tr_feat[i,j]]
            txt.append('%s 1 %s '% (j, j+1)+ ' '.join(str(e)+' ' for e in a))

np.savetxt('Mine_Graph_RML.txt', txt, fmt='%s')
