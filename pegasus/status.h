//
// Created by weiwen.cww on 19/1/2.
//
#pragma once

#include <algorithm>
#include <memory>
#include <string>

namespace gemini {

class Status {
 public:
  enum ErrorCode {
    kOk = 0,
    kArgumentError,
    kLogicError,
    kIndexOverflow,
    kNotFound,
    kDataLoss,
    kPermissionError,
    kInvalidFormat,
    kAlreadyExist,
    kNotImplemented,
    kVersionMismatch,
    kNotReady,
    kNetworkError,
    kTimeout,
    kServerFuncNotFound,
    kServerSerializeFailed,
    kServerDeserializeFailed,
    kClientSerializeFailed,
    kClientDeserializeFailed,
    kInternalError,
    kFileIOError,
    kIllegalStateError,
    kUnknown
  };

  Status() : state_(nullptr) {}
  ~Status() { delete state_; }

  Status(ErrorCode code, const std::string &msg) : state_(new State{code, msg}) {}

  Status(const Status &s) : state_(s.state_ == nullptr ? nullptr : new State(*s.state_)) {}
  void operator=(const Status &s) {
    if (state_) delete state_;
    state_ = s.state_ == nullptr ? nullptr : new State(*s.state_);
  }

  Status(Status &&s) : state_(s.state_) { s.state_ = nullptr; }
  void operator=(Status &&s) { std::swap(state_, s.state_); }

  bool IsOk() const { return state_ == nullptr; }

  ErrorCode Code() const { return state_ == nullptr ? kOk : state_->code; }

  std::string Msg() const { return state_ == nullptr ? "" : state_->msg; }

  bool operator==(const Status &x) const {
    return (state_ == nullptr && x.state_ == nullptr) ||
           (state_ != nullptr && x.state_ != nullptr && state_->code == x.state_->code && state_->msg == x.state_->msg);
  }
  bool operator!=(const Status &x) const { return !(*this == x); }

  std::string ToString() const {
    if (state_ == nullptr) {
      return "OK";
    } else {
      return "ErrorCode [" + std::to_string(Code()) + "]: " + Msg();
    }
  }

  // clang-format off
  static Status Ok() { return Status(); }
  static Status ArgumentError(const std::string &msg) { return Status(kArgumentError, msg); }
  static Status LogicError(const std::string &msg) { return Status(kLogicError, msg); }
  static Status IndexOverflow(const std::string &msg) { return Status(kIndexOverflow, msg); }
  static Status NotFound(const std::string &msg) { return Status(kNotFound, msg); }
  static Status DataLoss(const std::string &msg) { return Status(kDataLoss, msg); }
  static Status PermissionError(const std::string &msg) { return Status(kPermissionError, msg); }
  static Status InvalidFormat(const std::string &msg) { return Status(kInvalidFormat, msg); }
  static Status AlreadyExist(const std::string &msg) { return Status(kAlreadyExist, msg); }
  static Status NotImplemented(const std::string &msg) { return Status(kNotImplemented, msg); }
  static Status VersionMismatch(const std::string &msg) { return Status(kVersionMismatch, msg); }
  static Status NotReady(const std::string &msg) { return Status(kNotReady, msg); }
  static Status NetworkError(const std::string &msg) { return Status(kNetworkError, msg); }
  static Status Timeout(const std::string &msg) { return Status(kTimeout, msg); }
  static Status ServerFuncNotFound(const std::string &msg) { return Status(kServerFuncNotFound, msg); }
  static Status ServerSerializeFailed(const std::string &msg) { return Status(kServerSerializeFailed, msg); }
  static Status ServerDeserializeFailed(const std::string &msg) { return Status(kServerDeserializeFailed, msg); }
  static Status ClientSerializeFailed(const std::string &msg) { return Status(kClientSerializeFailed, msg); }
  static Status ClientDeserializeFailed(const std::string &msg) { return Status(kClientDeserializeFailed, msg); }
  static Status InternalError(const std::string &msg) { return Status(kInternalError, msg); }
  static Status FileIOError(const std::string &msg) { return Status(kFileIOError, msg); }
  static Status IllegalStateError(const std::string &msg) { return Status(kIllegalStateError, msg); }
  static Status Unknown(const std::string &msg) { return Status(kUnknown, msg); }
  // clang-format on

 private:
  struct State {
    ErrorCode code;
    std::string msg;
  };

  State *state_;
};
}  // namespace gemini

inline std::ostream &operator<<(std::ostream &os, const gemini::Status &status) {
  os << status.ToString();
  return os;
}

#define CHECK_STATUS_RETURN(STATUS, VALUE) \
  do {                                     \
    gemini::Status __st__ = (STATUS);      \
    if (!__st__.IsOk()) {                  \
      return (VALUE);                      \
    }                                      \
  } while (0)

#define CHECK_STATUS_INTERNAL(STATUS, MSG)       \
  do {                                           \
    gemini::Status __st__ = (STATUS);            \
    if (!__st__.IsOk()) {                        \
      return gemini::Status::InternalError(__st__.Msg() + "->" + MSG); \
    }                                            \
  } while (0)

#define CHECK_STATUS_PRINT_RETURN(STATUS, VALUE)                              \
  do {                                                                        \
    gemini::Status __st__ = (STATUS);                                         \
    if (!__st__.IsOk()) {                                                     \
      std::cout << __FILE__ << "@" << __LINE__ << ":" << __st__ << std::endl; \
      return (VALUE);                                                         \
    }                                                                         \
  } while (0)

#define CHECK_STATUS(STATUS)       \
  do {                             \
    const auto &__st__ = (STATUS); \
    if (!__st__.IsOk()) {          \
      return __st__;               \
    }                              \
  } while (0)

#define CHECK_THEN_PRINT(STATUS, MSG)                 \
  do {                                                \
    const auto &__st__ = (STATUS);                    \
    if (!__st__.IsOk()) {                             \
      std::cerr << MSG << " " << __st__ << std::endl; \
    }                                                 \
  } while (0)

#define CHECK_BOOL(BOOL, STATUS) \
  do {                           \
    if ((BOOL)) {                \
      return (STATUS);           \
    }                            \
  } while (0)
