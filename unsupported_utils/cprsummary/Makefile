CC = gcc
CFLAGS = -UDEBUG -g
OBJS = cpr2summaries.o
LDFLAGS = -lm

cpr2summaries: $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

clean:
	$(RM) $(OBJS) cpr2summaries
