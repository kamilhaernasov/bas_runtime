class runtime 
{
public:
    explicit runtime();
    ~runtime();

    bool start();
    bool stop();
private:
    bool pre();
    bool cycle();
    bool post();
}