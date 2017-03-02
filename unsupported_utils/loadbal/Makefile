CC = gcc
CFLAGS = -UDEBUG -g
OBJS = loadbal.o
LDFLAGS = -lm

loadbal: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) loadbal
