class Node:
    def __init__(self,val,nxt=None):

        self.val=val
        self.next=nxt


    
class link_list:
    def __init__(self,val):
        self.node=Node(val)
        self.curr=self.node
        self.head=self.node
    
    def append(self,val):
        self.node2=Node(val)
        self.curr.next=self.node2
        self.curr=self.node2

    def insert(self,val,posn):
        self.node3 = Node(val)

        n = 0
        dummy = self.head
        while n != posn:
            dummy=self.next
        a = dummy.next.val
        b = dummy.next

        dummy.next.
        dummy.next = self.node3

        

        


l=link_list(1)
l.append(2)
l.append(3)
l.append(4)
l.append(5)
dummy=l.head
while dummy:
    print(dummy.val)
    print(dummy)
    print(dummy.next)
    dummy=dummy.next






