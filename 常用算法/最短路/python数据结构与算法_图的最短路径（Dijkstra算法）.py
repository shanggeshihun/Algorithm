G={1:{1:0,    2:1,    3:12},
     2:{2:0,    3:9,    4:3},
     3:{3:0,    5:5},
     4:{3:4,    4:0,    5:13,   6:15},
     5:{5:0,    6:4},
     6:{6:0}
	 }
def Dijkstra(G,v0,INF=999):
	book=set()
	minv=v0
	dis=dict((k,INF) for k in G.keys())
	dis[v0]=0
	while len(book)<len(G):
		book.add(minv)
		for w in G[minv]:
			if dis[minv]+G[minv][w]<dis[w]:
			# 更新已知距离
				dis[w]=dis[minv]+G[minv][w]
		new=INF
		
		for v in dis.keys():
			if v in book:
				continue
			if dis[v]<new:
				new=dis[v]
				minv=v
				
				