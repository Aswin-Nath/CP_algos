start=[0 for i in range(n)]
end=[0 for i in range(n)]
timer=[0]
def tour(cur,prev):
  start[cur]=timer[0]
  timer[0]+=1
  for nei in G[cur]:
      if nei!=prev:
          tour(nei,cur)
  end[cur]=timer[0]
