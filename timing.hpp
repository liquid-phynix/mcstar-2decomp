#pragma once
#include <sys/time.h>
#include <cstdio>

class TimeAcc {
  int m_times;
  long long int m_elapsed;
  timeval m_tval;
public:
  TimeAcc(): m_times(0), m_elapsed(), m_tval(){}
  void start(){ gettimeofday(&m_tval, NULL); }
  void stop(bool incr=true){
    timeval end;
    gettimeofday(&end, NULL);
    if(incr) m_times++;
    m_elapsed += (end.tv_sec - m_tval.tv_sec) * 1000 * 1000 + (end.tv_usec - m_tval.tv_usec); }
  void report_ms(const char* msg){
    float t = float(m_elapsed) / 1000.0;
    printf(msg, t);
  }
  void report_avg_ms(const char* msg){
    float t = m_times == 0 ? 0 : (float(m_elapsed) / m_times / 1000.0);
    printf(msg, t, m_times);
  }
  void reset(){ m_times = 0; m_elapsed = 0; }
};

