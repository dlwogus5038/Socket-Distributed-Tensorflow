import socket,select,threading,os;
import tensorflow as tf
 
host=socket.gethostname()
port=5964
addr=(host,port)
     
inputs=[]
fd_name={}

cur_person = 0
max_person = 4

ps_hosts = [str(host)+":2221",str(host)+":2222"]
worker_hosts = []
worker_str = ""

supervisor_ready = False
 
def conn():
    print ('Waiting server is runing...')
    ss=socket.socket()
    ss.bind(addr)
    ss.listen(max_person)
    
    return ss
 
def new_coming(ss):
    global cur_person
    client,add=ss.accept()
    print ('welcome ' + str(add))
    try:
        inputs.append(client)
        fd_name[client]=str(add[0])
        cur_person = cur_person + 1

        if cur_person == max_person:
            waiting_msg="Start running tensorflow server.. Please wait"
            return_flag = 1
        else:
            waiting_msg=str(cur_person) + " people are waiting. We still need " + 
                str(max_person - cur_person) + " more people."
            return_flag = 0
        print(waiting_msg)

        for other in inputs:
            if other!=ss:
                try:
                    other.send(waiting_msg.encode())
                except Exception as e:
                    print (e)

        return return_flag
        
    except Exception as e:
        print (e)

def run_new_server(ss):
    global ps_hosts
    global worker_hosts
    global worker_str

    port_num = 2223
    for tmp in inputs:
        if tmp is not ss:
            worker_hosts.append(fd_name[tmp] + ':' + str(port_num))
            worker_str = worker_str + fd_name[tmp] + ':' + str(port_num) + ','
            port_num = port_num + 1

    worker_str = worker_str[0:-1]
    
    system_command = "python3 server.py --ps_hosts=" + ps_hosts[0] + ',' + ps_hosts[1] +
        " --worker_hosts=" + worker_str +" --job_name=ps --task_index=1"
    os.system("gnome-terminal -e 'bash -c \"" + system_command + "; exec bash\"'")
    
def socket_server_run():
    global cur_person
    ss=conn()
    inputs.append(ss)
    
    while True:
        r,w,e=select.select(inputs,[],[])
        for temp in r:
            if temp is ss:
                return_flag = new_coming(ss)
                if return_flag == 1:
                    run_new_server(ss)
                    return ss
            else:
                disconnect=False
                try:
                    data= temp.recv(1024).decode()
                    if data == '':
                        cur_person = cur_person - 1
                        data="One person left. We still need " + str(max_person - cur_person) +
                             " more people."
                        disconnect=True
                except socket.error:
                    data=fd_name[temp]+ ' left'
                    disconnect=True
                    
                if disconnect:
                    inputs.remove(temp)
                    print (data)
                    for other in inputs:
                        if other!=ss and other!=temp:
                            try:
                                other.send(data.encode())
                            except Exception as e:
                                print (e)                   
                    del fd_name[temp]
                    
                else:
                    print (data)

def tensor_server_run(ss):
    print("Distributed Tensorflow Server is running..")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name="ps",
                             task_index=0)
    print("Cluster job: %s, task_index: %d, target: %s" % ("ps", 0, server.target))
    
    task_index = 0
    for other in inputs:
        if other!=ss:
            try:
                msg = "START!!/" + ps_hosts[0] + ',' + ps_hosts[1] + '/' + worker_str + '/' + str(task_index)
                task_index = task_index + 1
                other.send(msg.encode())
            except Exception as e:
                print (e)  

    server.join()

    print("-----------------------------------------------------------------")

def check_training_end(ss):
    global cur_person
    global supervisor_ready
    while True:
        r,w,e=select.select(inputs,[],[])
        if cur_person == 0:
            return
        for temp in r:
            if temp != ss:
                disconnect=False
                try:
                    data=temp.recv(1024).decode()
                    if data == '':
                        continue
                    if data == "END!!":
                        if temp == inputs[1]:
                            supervisor_ready = True
                        else:
                            disconnect=True
                        cur_person = cur_person - 1
                        num=str(cur_person)

                        if cur_person == 0:
                            print("End!!")
                        else:
                            print(str(cur_person) + " people left..")

                        if supervisor_ready == True:
                            try:
                                inputs[1].send(num.encode())
                            except Exception as e:
                                print (e) 

                except socket.error:
                    print("ERROR!!")
                    disconnect=True

    
    
if __name__=='__main__':
    ss = socket_server_run()
    t=threading.Thread(target=tensor_server_run,args=(ss,))
    t.start()
    t1=threading.Thread(target=check_training_end,args=(ss,))
    t1.start()
